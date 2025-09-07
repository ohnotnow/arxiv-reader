from __future__ import annotations

import asyncio
import uuid
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
import chromadb
from dateutil import parser as dateparser
from openai import OpenAI
from pypdf import PdfReader

# FastHTML for app + HTML building
from fasthtml.common import *  # type: ignore
from starlette.responses import FileResponse

# Silence HF tokenizers fork warnings (Chroma default embedder)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------- Configuration ----------
DATA_DIR = Path("papers")
STATE_FILE = Path("state.json")
FILE_INDEX = Path("file_index.json")

# Map user-friendly names to arXiv category codes
CATEGORIES: Dict[str, str] = {
    "Artificial Intelligence (cs.AI)": "cs.AI",
    "Computation and Language (cs.CL)": "cs.CL",
    "Computer Vision (cs.CV)": "cs.CV",
    "Machine Learning (cs.LG)": "cs.LG",
    "Robotics (cs.RO)": "cs.RO",
    "Information Retrieval (cs.IR)": "cs.IR",
    "Stat.ML (stat.ML)": "stat.ML",
}


# ---------- Utilities ----------
def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------- File index (UUID -> safe file path) ----------
def _load_file_index() -> Dict[str, Any]:
    if FILE_INDEX.exists():
        try:
            return json.loads(FILE_INDEX.read_text())
        except Exception:
            return {}
    return {}


def _save_file_index(idx: Dict[str, Any]) -> None:
    FILE_INDEX.write_text(json.dumps(idx, indent=2))


def _register_file(path: Path, kind: str, mime: Optional[str] = None) -> str:
    uid = str(uuid.uuid4())
    root = Path.cwd().resolve()
    p = path.resolve()
    rel = p.relative_to(root)
    idx = _load_file_index()
    idx[uid] = {
        "rel": str(rel),
        "kind": kind,
        "mime": mime or ("application/pdf" if kind == "pdf" else "text/plain; charset=utf-8"),
    }
    _save_file_index(idx)
    return uid


def _get_last_run(state: Dict[str, Any], category: str) -> Optional[str]:
    # Backward compatible: legacy format stored directly at top level
    if "last_run" in state and isinstance(state["last_run"], dict):
        return state["last_run"].get(category)
    return state.get(category)


def _set_last_run(state: Dict[str, Any], category: str, iso: str) -> None:
    if "last_run" not in state or not isinstance(state.get("last_run"), dict):
        state["last_run"] = {}
    state["last_run"][category] = iso


def _get_prefs(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("prefs", {}) if isinstance(state.get("prefs"), dict) else {}


def _set_prefs(state: Dict[str, Any], prefs: Dict[str, Any]) -> None:
    state["prefs"] = prefs


def _default_since() -> datetime:
    # Default to one week ago, UTC
    return datetime.now(timezone.utc) - timedelta(days=7)


def _human(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _slugify(name: str) -> str:
    name = re.sub(r"[\/:*?\"<>|]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180]


def _pdf_text(path: Path) -> str:
    try:
        with path.open("rb") as f:
            reader = PdfReader(f)
            texts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    texts.append(t)
            return "\n\n".join(texts)
    except Exception as e:
        return f"[Error extracting text: {e}]"


def arxiv_fetch_sync(category: str, since: datetime, interest: Optional[str] = None, max_results: int = 200) -> List[Dict[str, Any]]:
    """Fetch recent arXiv entries via the `arxiv` package and optionally filter by simple substring interest."""
    client = arxiv.Client()
    # We request by category and sort descending by submission date.
    search = arxiv.Search(query=f"cat:{category}", max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

    items: List[Dict[str, Any]] = []
    interest_lc = (interest or "").strip().lower()
    for r in client.results(search):
        pub_dt = r.published if isinstance(r.published, datetime) else None
        if not pub_dt:
            continue
        pub_dt = pub_dt.astimezone(timezone.utc)
        if pub_dt < since:
            # Because results are sorted by submitted date descending, we can break optionally.
            # But to be safe, continue and let the client handle page size.
            continue

        title = (r.title or "").strip()
        summary = (r.summary or "").strip()
        authors = ", ".join(a.name for a in getattr(r, "authors", []) if getattr(a, "name", None))
        entry_id = r.entry_id or ""
        # Extract arXiv id from entry_id URL
        arxid = None
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-]+)", entry_id)
        if m:
            arxid = m.group(1)
        else:
            # Fallback to r.get_short_id() if present
            short_id = getattr(r, "get_short_id", None)
            if callable(short_id):
                arxid = short_id()
        if not arxid:
            continue

        if interest_lc:
            blob = f"{title}\n{summary}".lower()
            if interest_lc not in blob:
                continue

        items.append(
            {
                "id": entry_id,
                "arxiv_id": arxid,
                "title": title,
                "authors": authors,
                "published": pub_dt,
                "summary": summary,
                "pdf_url": getattr(r, "pdf_url", None),
            }
        )

    return items


async def arxiv_fetch(category: str, since: datetime, interest: Optional[str] = None, max_results: int = 200) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(arxiv_fetch_sync, category, since, interest, max_results)


# ---------- Chroma narrowing ----------

def _chroma_client() -> chromadb.Client:
    # Persistent cache across runs
    path = ".chroma"
    try:
        return chromadb.PersistentClient(path=path)
    except Exception:
        # Fallback to in-memory if persistent not available
        return chromadb.Client()


def _ensure_cached_embeddings(coll: chromadb.api.models.Collection.CollectionAPI, items: List[Dict[str, Any]], category: Optional[str] = None) -> None:
    ids = [it["arxiv_id"] for it in items]
    if not ids:
        return
    try:
        existing = coll.get(ids=ids)
        existing_ids = existing.get("ids") or []
        # Some versions return a list; others may align order but include Nones. Normalize.
        if isinstance(existing_ids, list) and existing_ids and isinstance(existing_ids[0], list):
            flat = []
            for sub in existing_ids:
                flat.extend([x for x in sub if x])
            have = set(flat)
        else:
            have = set(x for x in existing_ids if x)
    except Exception:
        have = set()

    missing_items = [it for it in items if it["arxiv_id"] not in have]
    if missing_items:
        coll.add(
            ids=[it["arxiv_id"] for it in missing_items],
            documents=[f"{it['title']}\n\n{it.get('summary','')}" for it in missing_items],
            metadatas=[
                {
                    "title": it["title"],
                    "authors": it.get("authors", ""),
                    "published": (it.get("published").isoformat() if isinstance(it.get("published"), datetime) else str(it.get("published"))),
                    "published_ts": (it.get("published").timestamp() if isinstance(it.get("published"), datetime) else None),
                    "category": category or "",
                }
                for it in missing_items
            ],
        )


def narrow_with_chroma(items: List[Dict[str, Any]], interest: str, top_k: int = 50, category: Optional[str] = None) -> List[Dict[str, Any]]:
    if not items or not interest.strip():
        return items

    client = _chroma_client()
    cache = client.get_or_create_collection(name="arxiv_cache")
    # Ensure embeddings for current items are cached
    _ensure_cached_embeddings(cache, items, category)

    # Query entire cache, then filter to our current items by id, preserving score order
    want = min(max(len(items), top_k * 2), 2000)
    # Filter results to just this category if we have one
    where = {"category": category} if category else None
    q = cache.query(query_texts=[interest], n_results=want, where=where)  # type: ignore[arg-type]
    ordered_ids: List[str] = (q.get("ids") or [[ ]])[0]
    allowed = {it["arxiv_id"] for it in items}
    filtered_order = [i for i in ordered_ids if i in allowed]
    if not filtered_order:
        return items[:top_k]
    id_to_item = {it["arxiv_id"]: it for it in items}
    out = [id_to_item[i] for i in filtered_order if i in id_to_item]
    if len(out) < top_k:
        # pad with remaining items (original order) to reach top_k
        seen = set(filtered_order)
        out.extend([it for it in items if it["arxiv_id"] not in seen])
    return out[:top_k]


def openai_summarize(text: str, style: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[OPENAI_API_KEY is not set; skipping summarization.]"
    client = OpenAI(api_key=api_key)
    # Truncate to a reasonable length to control tokens
    snippet = text
    if len(snippet) > 100_000:
        snippet = snippet[:100_000]
    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": "You summarize academic PDFs.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following paper for this audience: '{style}'.\n\nReturn 6-10 bullet points covering: goal, method, data, key results, limitations, and why it matters.\n\nText begins:\n{snippet}",
                },
            ],
            max_output_tokens=800,
        )
        # Responses API returns output in a .output_text convenience property in some SDKs;
        # here we extract from choices[0].message.content if needed.
        # The python SDK exposes .output_text for convenience.
        return resp.output_text  # type: ignore[attr-defined]
    except Exception as e:
        return f"[OpenAI API error: {e}]"


# ---------- Web App (FastHTML) ----------

tailwind = Script(src="https://cdn.tailwindcss.com")

app, rt = fast_app(hdrs=(tailwind,))


def category_select(selected: str | None = None):
    opts = [Option(name, value=code, selected=(code == selected)) for name, code in CATEGORIES.items()]
    return Select(
        *opts,
        name="category",
        cls=(
            "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
            "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
        ),
    )


def paper_row(item: Dict[str, Any]) -> Any:
    pid = item["arxiv_id"]
    title = item["title"]
    authors = item["authors"]
    pub = _human(item["published"]) if isinstance(item["published"], datetime) else str(item["published"]) 

    summary = item.get("summary", "")
    return Div(
        Div(
            Input(
                type="checkbox",
                name="selected",
                value=pid,
                cls=(
                    "mr-3 h-5 w-5 shrink-0 rounded-sm border-2 border-slate-400 dark:border-slate-500 "
                    "bg-white dark:bg-slate-900 accent-emerald-600 focus:ring-2 focus:ring-emerald-500"
                ),
            ),
            Div(
                H3(title, cls="font-semibold text-lg"),
                P(f"Authors: {authors}", cls="text-sm text-slate-600 dark:text-slate-300"),
                P(f"Published: {pub}", cls="text-xs text-slate-500 dark:text-slate-400"),
                P(summary, cls="mt-2 text-sm"),
                cls="flex-1"
            ),
            cls="flex items-start gap-2"
        ),
        cls="p-4 border rounded-lg shadow-sm bg-white dark:bg-slate-800 dark:border-slate-700 w-full"
    )


@rt("/")
def index(category: str | None = None, interest: str | None = None, summary_style: str | None = None, use_embeddings: str | None = None, top_k: int | None = None):
    state = _load_state()
    prefs = _get_prefs(state)
    cat_code = category or prefs.get("category") or next(iter(CATEGORIES.values()))
    last_run_iso = _get_last_run(state, cat_code)
    last_run = dateparser.parse(last_run_iso) if last_run_iso else _default_since()

    default_style = (
        summary_style
        or prefs.get(
            "summary_style",
            "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
        )
    )
    default_use_emb = (use_embeddings if use_embeddings is not None else prefs.get("use_embeddings", "on"))
    default_top_k = (top_k if top_k is not None else int(prefs.get("top_k", 50)))
    interest_value = interest if interest is not None else prefs.get("interest", "")

    return Titled(
        "arXiv Helper",
        Div(
            Div(
                H1("arXiv Helper", cls="text-2xl font-bold mb-2"),
                P("Find new papers by category since last run.", cls="text-slate-600 dark:text-slate-300 mb-4"),
                Div(
                    Form(
                        Div(
                            Label("Category", cls="font-medium"),
                            category_select(cat_code),
                            cls="flex flex-col gap-1"
                        ),
                        Div(
                            Label("Specific interest (optional)", cls="font-medium"),
                            Input(
                                name="interest",
                                placeholder="e.g. retrieval-augmented generation",
                                value=interest_value,
                                cls=(
                                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                                ),
                            ),
                            cls="flex flex-col gap-1"
                        ),
                        Div(
                            Label("Summary style", cls="font-medium"),
                            Textarea(
                                default_style,
                                name="summary_style",
                                rows=4,
                                cls=(
                                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                                ),
                            ),
                            cls="flex flex-col gap-1"
                        ),
                        Div(
                            Label("Narrow results with embeddings", cls="font-medium"),
                            Div(
                                Input(
                                    type="checkbox",
                                    name="use_embeddings",
                                    checked=(default_use_emb != "off"),
                                    cls=(
                                        "h-4 w-4 mr-2 shrink-0 rounded-sm border-2 border-slate-400 dark:border-slate-500 "
                                        "bg-white dark:bg-slate-900 accent-blue-600 focus:ring-2 focus:ring-blue-500"
                                    ),
                                ),
                                Label("Use semantic filter", cls="text-sm text-slate-700 dark:text-slate-300"),
                                cls="flex items-center"
                            ),
                            cls="flex flex-col gap-1"
                        ),
                        Div(
                            Label("Top K results", cls="font-medium"),
                            Input(type="number", name="top_k", value=str(default_top_k), min="5", max="200", cls=(
                                "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                                "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                            )),
                            cls="flex flex-col gap-1"
                        ),
                        Div(
                            Button("Fetch", type="submit", formaction="/fetch", formmethod="post", cls="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"),
                            cls="mt-2"
                        ),
                        cls="space-y-4"
                    ),
                    Div(
                        P(f"Last run for {cat_code}: {_human(last_run)}", cls="text-sm text-slate-600 dark:text-slate-300"),
                        cls="mt-4"
                    ),
                    cls="bg-white dark:bg-slate-800 dark:border-slate-700 p-4 rounded-lg border"
                ),
                cls="container mx-auto max-w-3xl p-4"
            ),
            cls="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-900 dark:text-slate-100"
        ),
    )


@rt("/fetch")
async def fetch(category: str, interest: str = "", summary_style: str = "", use_embeddings: str = "on", top_k: str = "50"):
    state = _load_state()
    since_iso = _get_last_run(state, category)
    since = dateparser.parse(since_iso) if since_iso else _default_since()

    # If using embeddings, fetch without simple substring filter to avoid over-pruning
    items = await arxiv_fetch(category, since, None if (use_embeddings != "off" and interest) else (interest or None))

    # Update state last-run now
    _set_last_run(state, category, datetime.now(timezone.utc).isoformat())
    # Save user prefs for next time
    prefs = _get_prefs(state)
    prefs.update(
        {
            "category": category,
            "interest": interest,
            "use_embeddings": ("off" if use_embeddings == "off" else "on"),
            "top_k": int(top_k or "50"),
            "summary_style": summary_style,
        }
    )
    _set_prefs(state, prefs)
    _save_state(state)

    # Optional narrowing via Chroma embeddings
    narrowed_items = items
    narrowing_error = None
    try:
        if (use_embeddings != "off") and interest:
            k = max(5, min(200, int(top_k or "50")))
            narrowed_items = narrow_with_chroma(items, interest, k, category)
    except Exception as e:
        narrowing_error = str(e)
        narrowed_items = items
    # Server-side debug log
    print(
        "[DEBUG] fetch",
        {
            "fetched": len(items),
            "semantic": "on" if use_embeddings != "off" else "off",
            "top_k": top_k,
            "narrowed": len(narrowed_items),
            "interest": interest,
            "error": narrowing_error,
        },
        flush=True,
    )

    results = [paper_row(it) for it in narrowed_items]
    if not results:
        results = [Div("No new papers found for the chosen filters.", cls="p-4 text-slate-600 dark:text-slate-300")]  # type: ignore[assignment]

    return Titled(
        "ArXiv Results",
        Div(
            Div(
                H1("Results", cls="text-2xl font-bold mb-4"),
                Form(
                    Div(
                        Input(type="hidden", name="category", value=category),
                        Input(type="hidden", name="interest", value=interest),
                        Input(type="hidden", name="use_embeddings", value=("on" if use_embeddings != "off" else "off")),
                        Input(type="hidden", name="top_k", value=str(top_k)),
                        Textarea(
                            summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
                            name="summary_style",
                            cls="hidden",
                        ),
                    ),
                    Div(*results, cls="grid grid-cols-1 gap-4"),
                    Div(
                        Div(
                            Button(
                                "Download selected and summarize",
                                type="submit",
                                formaction="/download",
                                formmethod="post",
                                cls="h-10 inline-flex items-center px-4 bg-emerald-600 text-white rounded hover:bg-emerald-700",
                            ),
                            A(
                                "Back",
                                href="/",
                                cls="h-10 inline-flex items-center px-4 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded",
                            ),
                            cls="flex items-center gap-3",
                        ),
                        Div(
                            Input(
                                type="number",
                                name="previous_days",
                                value="7",
                                min="1",
                                max="60",
                                cls="w-24 border rounded p-2 border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900",
                            ),
                            Button(
                                "Show previous matches",
                                type="submit",
                                formaction="/previous",
                                formmethod="post",
                                cls="px-4 py-2 bg-slate-600 text-white rounded hover:bg-slate-700",
                            ),
                            cls="flex items-center gap-2 mt-3",
                        ),
                        cls="mt-4",
                    ),
                    cls="",
                ),
                Div(
                    P(
                        f"Debug: fetched={len(items)} use_embeddings={'on' if use_embeddings!='off' else 'off'} top_k={top_k} narrowed={len(narrowed_items)} interest='{interest}'",
                        cls="text-xs text-slate-500 dark:text-slate-400 mt-2",
                    ),
                    P(f"Chroma error: {narrowing_error}", cls="text-xs text-red-500") if narrowing_error else None,
                ),
                cls="container mx-auto max-w-3xl p-4 bg-white dark:bg-slate-800 dark:border-slate-700 rounded-lg border",
            ),
            cls="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-900 dark:text-slate-100",
        ),
    )


def _download_pdf_via_arxiv(arxid: str, dest: Path) -> None:
    client = arxiv.Client()
    # Fetch the specific paper by id and use helper to download
    result = next(client.results(arxiv.Search(id_list=[arxid])))
    # arxiv library writes the file; supply dirpath and filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.download_pdf(dirpath=str(dest.parent), filename=dest.name)


def _serve_by_uuid(uid: str, expected_kind: str) -> Response:
    idx = _load_file_index()
    meta = idx.get(uid)
    if not isinstance(meta, dict):
        return Response("Not Found", status_code=404)
    if meta.get("kind") != expected_kind:
        return Response("Not Found", status_code=404)
    rel = meta.get("rel")
    if not rel:
        return Response("Not Found", status_code=404)
    root = Path.cwd().resolve()
    p = (root / rel).resolve()
    try:
        p.relative_to(root)
    except Exception:
        return Response("Forbidden", status_code=403)
    if not p.exists() or not p.is_file():
        return Response("Not Found", status_code=404)
    mime = meta.get("mime") or ("application/pdf" if expected_kind == "pdf" else "text/plain; charset=utf-8")
    return FileResponse(str(p), media_type=mime, filename=p.name)


@rt("/files/pdf/{uid}")
def file_pdf(uid: str):
    return _serve_by_uuid(uid, "pdf")


@rt("/files/text/{uid}")
def file_text(uid: str):
    return _serve_by_uuid(uid, "text")


@rt("/files/summary/{uid}")
def file_summary(uid: str):
    return _serve_by_uuid(uid, "summary")
@rt("/download_file")
def download_file(path: str):
    root = Path.cwd().resolve()
    p = (root / Path(path.lstrip("/"))).resolve()
    try:
        p.relative_to(root)
    except Exception:
        print(f"[DEBUG] download_file forbidden path={path} abs={p}", flush=True)
        return Response("Forbidden", status_code=403)
    if not p.exists() or not p.is_file():
        print(f"[DEBUG] download_file 404 path={path} abs={p}", flush=True)
        return Response("Not Found", status_code=404)
    print(f"[DEBUG] download_file OK path={path} abs={p}", flush=True)
    return FileResponse(str(p))


@rt("/download")
async def download(request: Request, category: str = "", interest: str = "", summary_style: str = ""):
    form = await request.form()
    # Multiple checkbox values under key 'selected'
    selected_ids = form.getlist("selected")  # type: ignore[attr-defined]
    if not selected_ids:
        return Titled(
            "No Selection",
            Div(
                P("No papers selected."),
                A("Back", href="/", cls="inline-block mt-2 px-3 py-2 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded"),
                cls="p-4"
            )
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = DATA_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results_ui: List[Any] = []
    meta_client = arxiv.Client()
    for arxid in selected_ids:
        safe_name = _slugify(arxid)
        pdf_path = out_dir / f"{safe_name}.pdf"
        txt_path = out_dir / f"{safe_name}.txt"
        sum_path = out_dir / f"{safe_name}.summary.txt"

        try:
            # Use arxiv package to download the PDF
            await asyncio.to_thread(_download_pdf_via_arxiv, arxid, pdf_path)
            # Fetch metadata for nicer display
            title = arxid
            authors = ""
            published_str = ""
            try:
                r = next(meta_client.results(arxiv.Search(id_list=[arxid])))
                title = (r.title or title).strip()
                authors = ", ".join(a.name for a in getattr(r, "authors", []) if getattr(a, "name", None))
                if isinstance(r.published, datetime):
                    published_str = _human(r.published)
            except Exception:
                pass
            text = _pdf_text(pdf_path)
            txt_path.write_text(text)
            summary = openai_summarize(text, summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding")
            sum_path.write_text(summary)
            # Post-write sanity check
            pdf_ok = pdf_path.exists()
            txt_ok = txt_path.exists()
            sum_ok = sum_path.exists()
            print(
                "[DEBUG] saved",
                {
                    "pdf": str(pdf_path),
                    "txt": str(txt_path),
                    "sum": str(sum_path),
                    "exists": {"pdf": pdf_ok, "txt": txt_ok, "sum": sum_ok},
                },
                flush=True,
            )
            pdf_uid = _register_file(pdf_path, "pdf", "application/pdf") if pdf_ok else None

            results_ui.append(
                Div(
                    Div(
                        H3(title, cls="text-xl font-semibold text-slate-900 dark:text-slate-100"),
                        P(f"Authors: {authors}", cls="text-sm text-slate-600 dark:text-slate-300") if authors else None,
                        P(f"Published: {published_str}", cls="text-xs text-slate-500 dark:text-slate-400") if published_str else None,
                        cls="mb-2"
                    ),
                    Div(summary, cls="mt-2 whitespace-pre-wrap leading-relaxed text-[0.95rem]"),
                    Div(
                        A(
                            "Open PDF",
                            href=(f"/files/pdf/{pdf_uid}" if pdf_uid else f"https://arxiv.org/pdf/{arxid}.pdf"),
                            target="_blank",
                            cls="inline-flex items-center text-sm font-medium text-indigo-600 dark:text-indigo-300 hover:underline",
                        ),
                        cls="mt-3"
                    ),
                    cls="p-5 border rounded-xl shadow-sm bg-white dark:bg-slate-800 dark:border-slate-700"
                )
            )
        except Exception as e:
            results_ui.append(Div(H3(f"{arxid}"), P(f"Error: {e}", cls="text-red-600"), cls="p-3 border rounded"))

    return Titled(
        "Download + Summaries",
        Div(
            H1("Download + Summaries", cls="text-2xl font-bold mb-3"),
            P(f"Saved under {out_dir}", cls="text-sm text-slate-600 dark:text-slate-300"),
            Div(*results_ui, cls="mt-4 space-y-3"),
            Div(
                A("Back", href="/", cls="inline-block mt-4 px-4 py-2 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded"),
            ),
            cls="container mx-auto max-w-3xl p-4"
        ),
    )


@rt("/previous")
async def previous(category: str, interest: str = "", use_embeddings: str = "on", top_k: str = "50", previous_days: str = "7", summary_style: str = ""):
    # Look back over past N days in the Chroma cache and return top-k matches
    k = max(5, min(200, int(top_k or "50")))
    days = max(1, min(60, int(previous_days or "7")))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    cutoff_ts = cutoff.timestamp()

    client = _chroma_client()
    cache = client.get_or_create_collection(name="arxiv_cache")

    # Build category filter only; apply time filter in Python to support existing cache without numeric timestamps
    where: Optional[Dict[str, Any]] = ({"category": category} if category else None)

    # If no interest or semantic filter off, do a recency-only fallback
    if not interest.strip() or use_embeddings == "off":
        q = cache.get(where=where)
        ids = q.get("ids") or []
        mds = q.get("metadatas") or []
        docs = q.get("documents") or []
        # Flatten possible nested lists
        if ids and isinstance(ids[0], list):
            flat = []
            for sub in ids:
                flat.extend(sub)
            ids = flat
            mds = [x for sub in (mds or []) for x in (sub or [])]
            docs = [x for sub in (docs or []) for x in (sub or [])]
        # Filter by cutoff and sort by published desc
        items_all: List[Dict[str, Any]] = []
        for i, md, doc in zip(ids, mds, docs):
            md = md or {}
            pub = None
            ts = md.get("published_ts")
            if isinstance(ts, (int, float)):
                try:
                    pub = datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    pub = None
            if not pub:
                try:
                    pub = dateparser.parse((md or {}).get("published", ""))
                except Exception:
                    pub = None
            items_all.append(
                {
                    "id": f"https://arxiv.org/abs/{i}",
                    "arxiv_id": i,
                    "title": md.get("title", ""),
                    "authors": md.get("authors", ""),
                    "published": pub or cutoff,
                    "summary": doc or "",
                }
            )
        items_all = [x for x in items_all if (x["published"] or cutoff) >= cutoff]
        items_all.sort(key=lambda x: x["published"] or cutoff, reverse=True)
        narrowed_items = items_all[:k]
    else:
        # Query top matches by interest, then filter by time in Python
        want = max(k * 5, k)
        res = cache.query(query_texts=[interest], n_results=want, where=where)
        ids = (res.get("ids") or [[]])[0]
        mds = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        items_all: List[Dict[str, Any]] = []
        for i, md, doc in zip(ids, mds, docs):
            md = md or {}
            pub = None
            ts = md.get("published_ts")
            if isinstance(ts, (int, float)):
                try:
                    pub = datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    pub = None
            if not pub:
                try:
                    pub = dateparser.parse(md.get("published", ""))
                except Exception:
                    pub = None
            if pub and pub.timestamp() < cutoff_ts:
                continue
            items_all.append(
                {
                    "id": f"https://arxiv.org/abs/{i}",
                    "arxiv_id": i,
                    "title": md.get("title", ""),
                    "authors": md.get("authors", ""),
                    "published": pub or cutoff,
                    "summary": doc or "",
                }
            )
        narrowed_items = items_all[:k]

    results = [paper_row(it) for it in narrowed_items]
    if not results:
        results = [Div("No cached matches in the selected window.", cls="p-4 text-slate-600 dark:text-slate-300")]  # type: ignore[assignment]

    return Titled(
        "arXiv Helper",
        Div(
            Div(
                H1("Previous Matches", cls="text-2xl font-bold mb-4"),
                Form(
                    Div(
                        Input(type="hidden", name="category", value=category),
                        Input(type="hidden", name="interest", value=interest),
                        Input(type="hidden", name="use_embeddings", value=("on" if use_embeddings != "off" else "off")),
                        Input(type="hidden", name="top_k", value=str(k)),
                        Input(type="hidden", name="previous_days", value=str(days)),
                        Textarea(summary_style or "", name="summary_style", cls="hidden"),
                    ),
                    Div(*results, cls="grid grid-cols-1 gap-4"),
                    Div(
                        Button(
                            "Download selected and summarize",
                            type="submit",
                            formaction="/download",
                            formmethod="post",
                            cls="h-10 inline-flex items-center px-4 bg-emerald-600 text-white rounded hover:bg-emerald-700",
                        ),
                        A("Back", href="/", cls="h-10 inline-flex items-center px-4 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded"),
                        cls="flex items-center gap-3"
                    ),
                        cls="mt-4"
                    ),
                    cls=""
                ),
                Div(
                    P(
                        f"Debug: previous-days={days} top_k={k} interest='{interest}'",
                        cls="text-xs text-slate-500 dark:text-slate-400 mt-2"
                    ),
                ),
                cls="container mx-auto max-w-3xl p-4 bg-white dark:bg-slate-800 dark:border-slate-700 rounded-lg border"
            ),
            cls="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-900 dark:text-slate-100"
        ),
    )


# Removed legacy generic /files route in favor of UUID-based routes above


@app.on_event("startup")
async def _maybe_fresh_state() -> None:  # type: ignore[attr-defined]
    if os.getenv("FRESH") == "1":
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("[DEBUG] FRESH=1: removed state.json", flush=True)
        except Exception as e:
            print(f"[DEBUG] FRESH=1: could not remove state.json: {e}", flush=True)


if __name__ == "__main__":
    import uvicorn
    import sys

    # Ensure data dir exists
    DATA_DIR.mkdir(exist_ok=True)
    if "--fresh" in sys.argv:
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("--fresh: removed state.json")
        except Exception as e:
            print(f"--fresh: could not remove state.json: {e}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
