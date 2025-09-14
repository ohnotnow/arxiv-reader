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


# ---------- Session-scoped tracking (advances on shutdown) ----------
SESSION_ANCHOR: Dict[str, datetime] = {}
LATEST_SEEN: Dict[str, datetime] = {}
TOUCHED_CATEGORIES: set[str] = set()


def _get_session_anchor(category: str) -> datetime:
    if category in SESSION_ANCHOR:
        return SESSION_ANCHOR[category]
    state = _load_state()
    iso = _get_last_run(state, category)
    dt = dateparser.parse(iso) if iso else _default_since()
    SESSION_ANCHOR[category] = dt
    return dt


def _update_latest_seen(category: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    # Compute max published time from items
    latest: Optional[datetime] = None
    for it in items:
        pub = it.get("published")
        if isinstance(pub, datetime):
            latest = pub if (latest is None or pub > latest) else latest
    if latest is not None:
        prev = LATEST_SEEN.get(category)
        if (prev is None) or (latest > prev):
            LATEST_SEEN[category] = latest
    TOUCHED_CATEGORIES.add(category)


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


# ---------- History helpers ----------
def _get_history(state: Dict[str, Any]) -> Dict[str, Any]:
    hist = state.get("history")
    return hist if isinstance(hist, dict) else {}


def _set_history(state: Dict[str, Any], history: Dict[str, Any]) -> None:
    state["history"] = history


def _normalize_interest(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _normalize_style(value: str) -> str:
    # Preserve case in storage; normalize only for dedupe
    return re.sub(r"\s+", " ", (value or "").strip())


# ---------- Named Styles (saved styles with title + body) ----------
def _get_styles(state: Dict[str, Any]) -> Dict[str, Any]:
    styles = state.get("styles")
    if isinstance(styles, dict) and isinstance(styles.get("items"), list):
        return styles
    return {"items": []}


def _set_styles(state: Dict[str, Any], styles: Dict[str, Any]) -> None:
    if not isinstance(styles, dict):
        styles = {"items": []}
    items = styles.get("items")
    if not isinstance(items, list):
        styles["items"] = []
    state["styles"] = styles


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _find_style_index(styles: Dict[str, Any], title: str) -> Optional[int]:
    items = styles.get("items")
    if not isinstance(items, list):
        return None
    key = _normalize_title(title)
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        t = _normalize_title(str(it.get("title", "")))
        if t == key:
            return i
    return None


def _upsert_style(state: Dict[str, Any], title: str, body: str) -> None:
    styles = _get_styles(state)
    items = styles.get("items")
    if not isinstance(items, list):
        items = []
        styles["items"] = items
    idx = _find_style_index(styles, title)
    now_iso = datetime.now(timezone.utc).isoformat()
    if idx is not None:
        it = items[idx] if idx < len(items) else {}
        it = {**it, "title": title, "body": body, "ts": now_iso, "count": int(it.get("count", 0) or 0) + 1}
        items[idx] = it
    else:
        items.insert(0, {"title": title, "body": body, "ts": now_iso, "count": 1})
    _set_styles(state, styles)


def _delete_style(state: Dict[str, Any], title: str) -> bool:
    styles = _get_styles(state)
    items = styles.get("items")
    if not isinstance(items, list):
        return False
    idx = _find_style_index(styles, title)
    if idx is None:
        return False
    try:
        items.pop(idx)
    except Exception:
        return False
    _set_styles(state, styles)
    return True


def _push_history(history: Dict[str, Any], kind: str, value: str, *, category: Optional[str] = None, cap: int = 25) -> None:
    if not value or not value.strip():
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    if kind == "interests":
        key = _normalize_interest(value)
        lst = history.get("interests")
        if not isinstance(lst, list):
            lst = []
        # Find existing
        idx = None
        for i, it in enumerate(lst):
            if _normalize_interest(it.get("value", "")) == key and (category or "") == (it.get("category") or ""):
                idx = i
                break
        if idx is not None:
            it = lst.pop(idx)
            it["ts"] = now_iso
            it["count"] = int(it.get("count", 0) or 0) + 1
            lst.insert(0, it)
        else:
            lst.insert(0, {"value": value, "ts": now_iso, "category": category or "", "count": 1})
        # Cap length
        history["interests"] = lst[: max(1, cap)]
    elif kind == "summary_styles":
        key = _normalize_style(value)
        lst = history.get("summary_styles")
        if not isinstance(lst, list):
            lst = []
        idx = None
        for i, it in enumerate(lst):
            if _normalize_style(it.get("value", "")) == key:
                idx = i
                break
        if idx is not None:
            it = lst.pop(idx)
            it["ts"] = now_iso
            it["count"] = int(it.get("count", 0) or 0) + 1
            lst.insert(0, it)
        else:
            lst.insert(0, {"value": value, "ts": now_iso, "count": 1})
        history["summary_styles"] = lst[: max(1, cap)]


def _recent_interests(history: Dict[str, Any], category: Optional[str], limit: int = 10) -> List[str]:
    lst = history.get("interests")
    if not isinstance(lst, list):
        return []
    key_seen: set[str] = set()
    out: List[str] = []
    # Category-first
    wanted_cat = category or ""
    for phase in (True, False):
        for it in lst:
            if not isinstance(it, dict):
                continue
            if phase and (it.get("category") or "") != wanted_cat:
                continue
            if (not phase) and (it.get("category") or "") == wanted_cat:
                continue
            val = it.get("value", "")
            key = _normalize_interest(val)
            if val and key not in key_seen:
                key_seen.add(key)
                out.append(val)
                if len(out) >= limit:
                    return out
    return out


def _recent_styles(history: Dict[str, Any], limit: int = 10) -> List[str]:
    lst = history.get("summary_styles")
    if not isinstance(lst, list):
        return []
    key_seen: set[str] = set()
    out: List[str] = []
    for it in lst:
        if not isinstance(it, dict):
            continue
        val = it.get("value", "")
        key = _normalize_style(val)
        if val and key not in key_seen:
            key_seen.add(key)
            out.append(val)
            if len(out) >= limit:
                break
    return out


def _truncate_label(text: str, max_len: int = 80) -> str:
    t = text or ""
    return t if len(t) <= max_len else (t[: max_len - 1] + "…")

def _default_since() -> datetime:
    # Default to one week ago, UTC
    return datetime.now(timezone.utc) - timedelta(days=7)


def _human(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _slugify(name: str) -> str:
    name = re.sub(r"[\/:*?\"<>|]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180]


def _css_id(name: str) -> str:
    """Return a CSS-selector-safe id segment derived from ``name``.

    Replaces whitespace with dashes and collapses any non [a-zA-Z0-9_-]
    characters into single dashes so it can be safely used with
    document.querySelector without escaping (e.g., dots in arXiv IDs).
    """
    s = (name or "").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-zA-Z0-9_-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:180]


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


# ---------- Stable per-paper cache ----------
def _paper_cache_dir(arxid: str) -> Path:
    # Use versioned arXiv id if provided to avoid clobbering when papers update
    safe = _slugify(arxid)
    p = DATA_DIR / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


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
            # Results are sorted by submitted date descending: safe to stop scanning.
            break

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


def narrow_with_chroma(items: List[Dict[str, Any]], interest: str, top_k: int = 10, category: Optional[str] = None) -> List[Dict[str, Any]]:
    if not items or not interest.strip():
        return items

    client = _chroma_client()
    cache = client.get_or_create_collection(name="arxiv_cache")
    # Ensure embeddings for current items are cached
    _ensure_cached_embeddings(cache, items, category)

    # Query entire cache, then filter to our current items by id, preserving score order
    # Query a small multiple of top_k to limit payload sizes
    want = min(max(top_k * 3, top_k), 300)
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


def openai_summarize(
    text: str,
    style: str,
    verbosity: str = "medium",
    reasoning: str = "medium",
    title: Optional[str] = None,
) -> str:
    """Summarize text via OpenAI with a user-provided style.

    If the user style does not already mention 'markdown' (case-insensitive),
    append a small instruction asking for clear, well-structured Markdown.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[OPENAI_API_KEY is not set; skipping summarization.]"
    client = OpenAI(api_key=api_key)
    # Truncate to a reasonable length to control tokens
    snippet = text
    if len(snippet) > 100_000:
        snippet = snippet[:100_000]
    # Build user prompt: if a style is provided, use it; otherwise use a
    # sensible structured default. Avoid imposing structure when the user
    # gave a style, per product guidance.
    style_clean = (style or "").strip()
    if style_clean:
        base_instructions = (
            "Summarize the following paper for this audience: "
            f"'{style_clean}'."
        )
    else:
        base_instructions = (
            "Summarize the following paper.\n\n"
            "Return 6-10 bullet points covering: goal, method, data, key results, "
            "limitations, and why it matters."
        )

    # Title handling: omit title to avoid duplication in UI
    base_instructions += (
        "\n\nDo not include any title heading and do not repeat the paper title; "
        "begin directly with the summary content."
    )

    # Language guidance to avoid accidental non-English outputs when titles contain non-English words
    base_instructions += (
        "\n\nPlease write the summary in English unless the user has explicitly "
        "requested another language."
    )

    # Only append markdown guidance if the user hasn't already specified it
    if "markdown" not in (style_clean or "").lower():
        base_instructions += (
            "\n\nFormat the response as clear, well-structured Markdown. "
            "Use section headings and bullet lists where helpful; bold key terms. "
            "Avoid HTML and preambles — return only the Markdown summary."
        )

    # Add the text snippet
    base_instructions += f"\n\nText begins:\n{snippet}"

    # Build developer instructions to re-enable Markdown for reasoning models
    wants_markdown = "markdown" in (style_clean or "").lower()
    instructions_lines = [
        "Formatting re-enabled",
    ]
    if not wants_markdown:
        instructions_lines.append(
            "Format the response as clear, well-structured Markdown. "
            "Use section headings and bullet lists where helpful; bold key terms. "
            "Avoid HTML and preambles — return only the Markdown summary."
        )
    dev_instructions = "\n".join(instructions_lines)

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
                    "content": base_instructions,
                },
            ],
            instructions=dev_instructions,
            reasoning={"effort": reasoning},
            text={"verbosity": verbosity},
        )
        # Responses API returns output in a .output_text convenience property in some SDKs;
        # here we extract from choices[0].message.content if needed.
        # The python SDK exposes .output_text for convenience.
        return resp.output_text  # type: ignore[attr-defined]
    except Exception as e:
        return f"[OpenAI API error: {e}]"


# ---------- Web App (FastHTML) ----------

tailwind = Script(src="https://cdn.tailwindcss.com?plugins=typography")
htmx = Script(src="https://unpkg.com/htmx.org@1.9.12")
# Client-side Markdown rendering (UMD builds)
marked_js = Script(src="https://cdnjs.cloudflare.com/ajax/libs/marked/16.2.1/lib/marked.umd.min.js")
dompurify_js = Script(src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.2.6/purify.min.js")
# Inline script to convert [data-md] blocks from plain text to sanitized HTML
markdown_script = Script(
    """
    ;(function(){
      function renderMarkdown(root, force){
        try { console.log('[MD] renderMarkdown for', root); } catch(_e){}
        if(!window.marked || !window.DOMPurify) return;
        try { window.marked.setOptions({ breaks: true, gfm: true }); } catch(e){}
        var nodes = (root || document).querySelectorAll('[data-md]');
        try { console.log('[MD] nodes:', nodes.length, 'force=', !!force); } catch(_e){}
        try { console.log('[MD] nodes:', nodes.length); } catch(_e){}
        for (var i=0; i<nodes.length; i++){
          var el = nodes[i];
          if (el.dataset.mdRendered === '1' && !force) continue;
          var raw = el.textContent || '';
          var html = window.marked.parse(raw);
          el.innerHTML = window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
          el.dataset.mdRendered = '1';
          el.classList.add('prose','prose-slate','dark:prose-invert','max-w-none');
          var links = el.querySelectorAll('a[href^="http"]');
          for (var j=0; j<links.length; j++){ links[j].target = '_blank'; links[j].rel = 'noopener noreferrer'; }
        }
      }
      try { window.__renderMarkdown = renderMarkdown; } catch(_e){}
      document.addEventListener('DOMContentLoaded', function(){
        try { console.log('[MD] DOMContentLoaded'); } catch(_e){}
        renderMarkdown(document, false);
      });
      document.addEventListener('htmx:afterSwap', function(e){
        try { console.log('[MD] htmx:afterSwap'); } catch(_e){}
        var root = (e.detail && e.detail.target) || e.target || document;
        renderMarkdown(root, true);
      });
      document.addEventListener('htmx:afterSettle', function(e){
        try { console.log('[MD] htmx:afterSettle'); } catch(_e){}
        var root = (e.detail && e.detail.target) || e.target || document;
        renderMarkdown(root, true);
      });
    })();
    """
)

app, rt = fast_app(hdrs=(tailwind, htmx), pico=False)


def category_select(selected: str | None = None, last_checked_labels: Optional[Dict[str, str]] = None, select_attrs: Optional[Dict[str, Any]] = None, choices: Optional[Dict[str, str]] = None):
    opts = []
    mapping = choices or CATEGORIES
    for name, code in mapping.items():
        label = name
        if last_checked_labels and code in last_checked_labels:
            label = f"{name} (Last checked {last_checked_labels[code]})"
        opts.append(Option(label, value=code, selected=(code == selected)))
    attrs = select_attrs or {}
    return Select(
        *opts,
        name="category",
        **attrs,
        cls=(
            "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
            "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
        ),
    )


# ---------- Styles management (HTMX partials) ----------

@rt("/styles/save")
def styles_save(style_title_input: str = "", summary_style: str = ""):
    state = _load_state()
    title = (style_title_input or "").strip()
    body = (summary_style or "").strip()
    error = None
    saved_msg = None
    # Basic validation
    if not title or not body:
        error = "Both style title and style guide are required."
    elif len(title) > 60:
        error = "Style title is too long (max 60 characters)."
    else:
        _upsert_style(state, title, body)
        # Also remember as the preference for convenience
        prefs = _get_prefs(state)
        prefs["summary_style"] = body
        _set_prefs(state, prefs)
        _save_state(state)
        saved_msg = f"Saved style: {title}"

    # Re-render styles section partial
    saved_styles = _get_styles(state)
    saved_items: List[Dict[str, str]] = []
    if isinstance(saved_styles.get("items"), list):
        for it in saved_styles["items"]:
            if isinstance(it, dict) and (it.get("title") and it.get("body")):
                saved_items.append({"title": str(it["title"]), "body": str(it["body"])})

    def render_styles_section(current_title_val: str, current_body_val: str, error_msg: str | None = None, saved_msg: str | None = None):
        opts = [Option("Select a saved style…", value="", **{"data-title": ""})]
        for it in saved_items:
            t = it.get("title", "")
            b = it.get("body", "")
            sel = (t == current_title_val)
            opts.append(Option(t, value=b, selected=sel, **{"data-title": t, "title": t}))
        controls: List[Any] = []
        if saved_msg:
            controls.append(Div(saved_msg, cls="mb-2 p-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-sm"))
        if error_msg:
            controls.append(Div(error_msg, cls="mb-2 p-2 rounded bg-red-50 text-red-700 border border-red-200 text-sm"))
        controls.extend([
            Label("Saved styles", cls="text-sm text-slate-600 dark:text-slate-300"),
            Div(
                Select(
                    *opts,
                    onchange=(
                        "(function(sel){var opt=sel.selectedOptions[0];"
                        "var t=document.getElementById('style_title_input');"
                        "var b=document.getElementById('summary_style_input');"
                        "var s=document.getElementById('style_selected_title');"
                        "if(opt){ t.value=opt.dataset.title||''; b.value=opt.value||''; s.value=opt.dataset.title||''; }"
                        "})(this)"
                    ),
                    cls=(
                        "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                        "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                    ),
                ),
                Button(
                    "Delete",
                    type="button",
                    cls=(
                        "ml-2 h-10 px-3 rounded bg-red-600 text-white text-sm hover:bg-red-700 "
                        "disabled:opacity-50 disabled:cursor-not-allowed"
                    ),
                    **{
                        "hx-post": "/styles/delete",
                        "hx-target": "#styles_section",
                        "hx-swap": "outerHTML",
                        "hx-include": "#style_selected_title, #summary_style_input",
                        "hx-confirm": "Delete selected style?",
                    },
                ),
                cls="flex items-center"
            ),
            Input(type="hidden", id="style_selected_title", name="style_selected_title", value=title),
            Label("Style title", cls="text-sm text-slate-600 dark:text-slate-300"),
            Input(
                id="style_title_input",
                placeholder="e.g., ELI5, Executive summary, Reviewer notes",
                value=title,
                cls=(
                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                ),
            ),
            Label("Style guide", cls="text-sm text-slate-600 dark:text-slate-300"),
            Textarea(
                body,
                name="summary_style",
                id="summary_style_input",
                rows=4,
                placeholder="Describe how to write the summary…",
                cls=(
                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                ),
            ),
            Div(
                Button(
                    "Save / Update",
                    type="button",
                    cls=(
                        "mt-2 inline-flex items-center justify-center h-9 px-3 rounded bg-emerald-600 text-white "
                        "text-sm hover:bg-emerald-700"
                    ),
                    **{
                        "hx-post": "/styles/save",
                        "hx-target": "#styles_section",
                        "hx-swap": "outerHTML",
                        "hx-include": "#style_title_input, #summary_style_input",
                    },
                ),
                cls=""
            ),
        ])
        return Div(
            Label("Summary style", cls="font-medium"),
            Div(*controls, id="styles_section", cls="flex flex-col gap-1"),
            cls="flex flex-col gap-1"
        )

    return render_styles_section(title, body, error, saved_msg)


@rt("/styles/delete")
def styles_delete(style_selected_title: str = "", summary_style: str = ""):
    state = _load_state()
    title = (style_selected_title or "").strip()
    body = (summary_style or "").strip()
    error = None
    saved_msg = None
    if not title:
        error = "No style selected to delete."
    else:
        ok = _delete_style(state, title)
        if ok:
            _save_state(state)
            saved_msg = f"Deleted style: {title}"
        else:
            error = f"Style not found: {title}"

    saved_styles = _get_styles(state)
    saved_items: List[Dict[str, str]] = []
    if isinstance(saved_styles.get("items"), list):
        for it in saved_styles["items"]:
            if isinstance(it, dict) and (it.get("title") and it.get("body")):
                saved_items.append({"title": str(it["title"]), "body": str(it["body"])})

    def render_styles_section(current_title_val: str, current_body_val: str, error_msg: str | None = None, saved_msg: str | None = None):
        opts = [Option("Select a saved style…", value="", **{"data-title": ""})]
        for it in saved_items:
            t = it.get("title", "")
            b = it.get("body", "")
            sel = (t == current_title_val)
            opts.append(Option(t, value=b, selected=sel, **{"data-title": t, "title": t}))
        controls: List[Any] = []
        if saved_msg:
            controls.append(Div(saved_msg, cls="mb-2 p-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-sm"))
        if error_msg:
            controls.append(Div(error_msg, cls="mb-2 p-2 rounded bg-red-50 text-red-700 border border-red-200 text-sm"))
        controls.extend([
            Label("Saved styles", cls="text-sm text-slate-600 dark:text-slate-300"),
            Div(
                Select(
                    *opts,
                    onchange=(
                        "(function(sel){var opt=sel.selectedOptions[0];"
                        "var t=document.getElementById('style_title_input');"
                        "var b=document.getElementById('summary_style_input');"
                        "var s=document.getElementById('style_selected_title');"
                        "if(opt){ t.value=opt.dataset.title||''; b.value=opt.value||''; s.value=opt.dataset.title||''; }"
                        "})(this)"
                    ),
                    cls=(
                        "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                        "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                    ),
                ),
                Button(
                    "Delete",
                    type="button",
                    cls=(
                        "ml-2 h-10 px-3 rounded bg-red-600 text-white text-sm hover:bg-red-700 "
                        "disabled:opacity-50 disabled:cursor-not-allowed"
                    ),
                    **{
                        "hx-post": "/styles/delete",
                        "hx-target": "#styles_section",
                        "hx-swap": "outerHTML",
                        "hx-include": "#style_selected_title, #summary_style_input",
                        "hx-confirm": "Delete selected style?",
                    },
                ),
                cls="flex items-center"
            ),
            Input(type="hidden", id="style_selected_title", name="style_selected_title", value=("" if saved_msg else title)),
            Label("Style title", cls="text-sm text-slate-600 dark:text-slate-300"),
            Input(
                id="style_title_input",
                placeholder="e.g., ELI5, Executive summary, Reviewer notes",
                value=("" if saved_msg else title),
                cls=(
                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                ),
            ),
            Label("Style guide", cls="text-sm text-slate-600 dark:text-slate-300"),
            Textarea(
                ("" if saved_msg else body),
                name="summary_style",
                id="summary_style_input",
                rows=4,
                placeholder="Describe how to write the summary…",
                cls=(
                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                ),
            ),
            Div(
                Button(
                    "Save / Update",
                    type="button",
                    cls=(
                        "mt-2 inline-flex items-center justify-center h-9 px-3 rounded bg-emerald-600 text-white "
                        "text-sm hover:bg-emerald-700"
                    ),
                    **{
                        "hx-post": "/styles/save",
                        "hx-target": "#styles_section",
                        "hx-swap": "outerHTML",
                        "hx-include": "#style_title_input, #summary_style_input",
                    },
                ),
                cls=""
            ),
        ])
        return Div(
            Label("Summary style", cls="font-medium"),
            Div(*controls, id="styles_section", cls="flex flex-col gap-1"),
            cls="flex flex-col gap-1"
        )

    return render_styles_section("" if saved_msg else title, "" if saved_msg else body, error, saved_msg)


# ---------- User category list helpers ----------
def _get_user_categories(state: Dict[str, Any]) -> List[Dict[str, str]]:
    cats = state.get("categories")
    out: List[Dict[str, str]] = []
    if isinstance(cats, list):
        for it in cats:
            if isinstance(it, dict) and it.get("code") and it.get("label"):
                out.append({"label": str(it["label"]), "code": str(it["code"])})
    if out:
        return out
    # Fallback to defaults from CATEGORIES
    return [{"label": name.split(" (", 1)[0], "code": code} for name, code in CATEGORIES.items()]


def _set_user_categories(state: Dict[str, Any], items: List[Dict[str, str]]) -> None:
    # De-duplicate by code, keep order
    seen: set[str] = set()
    cleaned: List[Dict[str, str]] = []
    for it in items:
        code = (it.get("code") or "").strip()
        label = (it.get("label") or "").strip()
        if not code or not label:
            continue
        if code in seen:
            continue
        seen.add(code)
        cleaned.append({"label": label, "code": code})
    state["categories"] = cleaned


def _build_category_display_map(items: List[Dict[str, str]]) -> Dict[str, str]:
    # Ordered mapping: "Label (code)" -> code
    out: Dict[str, str] = {}
    for it in items:
        label = it.get("label") or ""
        code = it.get("code") or ""
        if not code:
            continue
        name = f"{label} ({code})" if label else code
        out[name] = code
    return out


def _load_category_cache() -> List[Dict[str, str]]:
    path = Path("data/arxiv_categories.json")
    try:
        if path.exists():
            data = json.loads(path.read_text())
            out: List[Dict[str, str]] = []
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict) and it.get("code") and it.get("label"):
                        out.append({
                            "code": str(it.get("code")),
                            "label": str(it.get("label")),
                            "group": str(it.get("group", "")),
                        })
            return out
    except Exception:
        return []
    return []


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if t]


def _fuzzy_search_categories(query: str, items: List[Dict[str, str]], limit: int = 10) -> List[Dict[str, str]]:
    q = _norm(query)
    if not q:
        return []
    qtoks = set(_tokenize(q))
    results: List[tuple[float, Dict[str, str]]] = []
    for it in items:
        code = it.get("code", "")
        label = it.get("label", "")
        group = it.get("group", "")
        code_l = _norm(code)
        label_l = _norm(label)
        group_l = _norm(group)
        # base score
        score = 0.0
        # direct matches on code
        if code_l.startswith(q):
            score += 60
        elif q in code_l:
            score += 40
        # label/group token overlap
        ltoks = set(_tokenize(label_l)) | set(_tokenize(group_l))
        inter = len(qtoks & ltoks)
        if inter:
            # weight by overlap and token proportion
            score += 10 * inter
            score += 30 * (inter / max(1, len(qtoks)))
        # substring in label
        if q in label_l:
            score += 20
        if score > 0:
            results.append((score, it))
    results.sort(key=lambda x: (-x[0], x[1].get("code", "")))
    return [it for _, it in results[:limit]]


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
                onchange=(
                    "(function(cb){var btn=document.getElementById('btn_download');"
                    "if(!btn) return; var any=document.querySelectorAll('input[name=selected]:checked').length>0;"
                    "btn.disabled=!any; btn.classList.toggle('opacity-50', !any); btn.classList.toggle('cursor-not-allowed', !any);"
                    "})(this)"
                ),
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
def index(category: str | None = None, interest: str | None = None, summary_style: str | None = None, use_embeddings: str | None = None, top_k: int | None = None, verbosity: str | None = None, reasoning: str | None = None):
    state = _load_state()
    prefs = _get_prefs(state)
    # User categories list (ordered); build display mapping
    user_cats = _get_user_categories(state)
    display_map = _build_category_display_map(user_cats)
    # Pick default category code from user list
    first_code = next(iter(display_map.values())) if display_map else next(iter(CATEGORIES.values()))
    cat_code = category or prefs.get("category") or first_code
    # Use session-stable anchor for this category
    last_run = _get_session_anchor(cat_code)

    default_style = (
        summary_style
        or prefs.get(
            "summary_style",
            "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
        )
    )
    default_use_emb = (use_embeddings if use_embeddings is not None else prefs.get("use_embeddings", "on"))
    default_top_k = (top_k if top_k is not None else int(prefs.get("top_k", 10)))
    interest_value = interest if interest is not None else prefs.get("interest", "")
    default_verbosity = verbosity or prefs.get("verbosity", "medium")
    default_reasoning = reasoning or prefs.get("reasoning", "medium")

    # Build recent history suggestions
    history = _get_history(state)
    recent_interests = _recent_interests(history, cat_code, limit=10)
    recent_styles = _recent_styles(history, limit=10)
    saved_styles = _get_styles(state)
    saved_items: List[Dict[str, str]] = []
    if isinstance(saved_styles.get("items"), list):
        for it in saved_styles["items"]:
            if isinstance(it, dict) and (it.get("title") and it.get("body")):
                saved_items.append({"title": str(it["title"]), "body": str(it["body"])})
    # Determine current title if the default style matches a saved body
    current_title = ""
    for it in saved_items:
        if it.get("body", "") == (default_style or ""):
            current_title = it.get("title", "")
            break

    def render_styles_section(current_title_val: str, current_body_val: str, error_msg: str | None = None, saved_msg: str | None = None):
        # Build a select of saved titles; use option value as body and data-title as title.
        opts = [Option("Select a saved style…", value="", **{"data-title": ""})]
        for it in saved_items:
            t = it.get("title", "")
            b = it.get("body", "")
            sel = (t == current_title_val)
            opts.append(Option(t, value=b, selected=sel, **{"data-title": t, "title": t}))
        controls = []
        if saved_msg:
            controls.append(Div(saved_msg, cls="mb-2 p-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-sm"))
        if error_msg:
            controls.append(Div(error_msg, cls="mb-2 p-2 rounded bg-red-50 text-red-700 border border-red-200 text-sm"))
        controls.extend([
            Label("Saved styles", cls="text-sm text-slate-600 dark:text-slate-300"),
            Div(
                Select(
                    *opts,
                    onchange=(
                        "(function(sel){var opt=sel.selectedOptions[0];"
                        "var t=document.getElementById('style_title_input');"
                        "var b=document.getElementById('summary_style_input');"
                        "var s=document.getElementById('style_selected_title');"
                        "if(opt){ t.value=opt.dataset.title||''; b.value=opt.value||''; s.value=opt.dataset.title||''; }"
                        "})(this)"
                    ),
                    cls=(
                        "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                        "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                    ),
                ),
                Button(
                    "Delete",
                    type="button",
                    cls=(
                        "ml-2 h-10 px-3 rounded bg-red-600 text-white text-sm hover:bg-red-700 "
                        "disabled:opacity-50 disabled:cursor-not-allowed"
                    ),
                    **{
                        "hx-post": "/styles/delete",
                        "hx-target": "#styles_section",
                        "hx-swap": "outerHTML",
                        "hx-include": "#style_selected_title, #summary_style_input",
                        "hx-confirm": "Delete selected style?",
                    },
                ),
                cls="flex items-center"
            ),
            Input(type="hidden", id="style_selected_title", name="style_selected_title", value=current_title_val),
            Label("Style title", cls="text-sm text-slate-600 dark:text-slate-300"),
            Input(
                id="style_title_input",
                placeholder="e.g., ELI5, Executive summary, Reviewer notes",
                value=current_title_val,
                cls=(
                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                ),
            ),
            Label("Style guide", cls="text-sm text-slate-600 dark:text-slate-300"),
            Textarea(
                current_body_val,
                name="summary_style",
                id="summary_style_input",
                rows=4,
                placeholder="Describe how to write the summary…",
                cls=(
                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                ),
            ),
            Div(
                Button(
                    "Save / Update",
                    type="button",
                    cls=(
                        "mt-2 inline-flex items-center justify-center h-9 px-3 rounded bg-emerald-600 text-white "
                        "text-sm hover:bg-emerald-700"
                    ),
                    **{
                        "hx-post": "/styles/save",
                        "hx-target": "#styles_section",
                        "hx-swap": "outerHTML",
                        "hx-include": "#style_title_input, #summary_style_input",
                    },
                ),
                cls=""
            ),
        ])
        return Div(
            Label("Summary style", cls="font-medium"),
            Div(*controls, id="styles_section", cls="flex flex-col gap-1"),
            cls="flex flex-col gap-1"
        )

    # Build category -> last checked label map for dropdown
    cat_last_checked: Dict[str, str] = {}
    for code in (display_map.values()):
        cat_last_checked[code] = _human(_get_session_anchor(code))

    return Html(
        Head(Title("arXiv Helper"), tailwind, htmx, marked_js, dompurify_js, markdown_script),
        Body(
            Main(
                Div(
                    P("Find new papers by category since last run.", cls="text-slate-600 dark:text-slate-300 mb-4"),
                Div(
                    Form(
                        Div(
                            Div(
                                Label("Category", cls="font-medium"),
                                A(
                                    "Manage categories",
                                    href="/categories",
                                    cls=(
                                        "inline-flex items-center justify-center h-10 px-3 "
                                        "bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 "
                                        "dark:hover:bg-slate-600 dark:text-slate-100 rounded text-sm"
                                    ),
                                ),
                                cls="flex items-center justify-between"
                            ),
                            category_select(
                                cat_code,
                                last_checked_labels=cat_last_checked,
                                select_attrs={
                                    "id": "category_select",
                                },
                                choices=display_map,
                            ),
                            cls="flex flex-col gap-1"
                        ),
                        Div(
                            Label("Specific interest (optional)", cls="font-medium"),
                            (Div(
                                Label("Recent interests", cls="text-sm text-slate-600 dark:text-slate-300"),
                                Select(
                                    Option("Select a recent interest…", value=""),
                                    *[Option(_truncate_label(s), value=s) for s in recent_interests],
                                    onchange="document.querySelector('#interest_input').value=this.value",
                                    cls=(
                                        "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                                        "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 mb-1"
                                    ),
                                ),
                                cls="flex flex-col gap-1"
                            ) if recent_interests else None),
                            Input(
                                name="interest",
                                id="interest_input",
                                placeholder="e.g. retrieval-augmented generation",
                                value=interest_value,
                                list="recent-interests",
                                cls=(
                                    "border rounded p-2 w-full border-slate-300 bg-white text-slate-900 "
                                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                                ),
                            ),
                            (Datalist(
                                *[Option(value=val) for val in recent_interests],
                                id="recent-interests",
                            ) if recent_interests else None),
                            cls="flex flex-col gap-1"
                        ),
                        render_styles_section(current_title, default_style),
                        Div(
                            Button(
                                Span("More settings ", cls=""),
                                Span("▼", cls="inline-block transition-transform duration-200", id="settings_chevron"),
                                type="button",
                                onclick=(
                                    "const panel=this.nextElementSibling; "
                                    "const chevron=this.querySelector('#settings_chevron'); "
                                    "const textSpan=this.querySelector('span:first-child'); "
                                    "const isHidden=panel.style.display==='none'; "
                                    "panel.style.display=isHidden?'block':'none'; "
                                    "textSpan.textContent=isHidden?'Less settings ':'More settings '; "
                                    "chevron.style.transform=isHidden?'rotate(180deg)':'rotate(0deg)';"
                                ),
                                cls="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 cursor-pointer text-sm font-normal bg-transparent border-none p-0 mb-2"
                            ),
                            Div(
                                Div(
                                    Div(
                                        Label("Top K results", cls="text-sm font-medium mb-1"),
                                        Input(type="number", name="top_k", value=str(default_top_k), min="5", max="200", cls=(
                                            "h-9 border rounded px-3 py-1.5 w-full border-slate-300 bg-white text-slate-900 "
                                            "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                                        )),
                                        cls="flex-1"
                                    ),
                                    Div(
                                        Label("Verbosity", cls="text-sm font-medium mb-1"),
                                        Select(
                                            Option("Low", value="low", selected=(default_verbosity == "low")),
                                            Option("Medium", value="medium", selected=(default_verbosity == "medium")),
                                            Option("High", value="high", selected=(default_verbosity == "high")),
                                            name="verbosity",
                                            cls=(
                                                "h-9 border rounded px-3 py-1.5 w-full border-slate-300 bg-white text-slate-900 "
                                                "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                                            ),
                                        ),
                                        cls="flex-1"
                                    ),
                                    Div(
                                        Label("Reasoning", cls="text-sm font-medium mb-1"),
                                        Select(
                                            Option("Minimal", value="minimal", selected=(default_reasoning == "minimal")),
                                            Option("Low", value="low", selected=(default_reasoning == "low")),
                                            Option("Medium", value="medium", selected=(default_reasoning == "medium")),
                                            Option("High", value="high", selected=(default_reasoning == "high")),
                                            name="reasoning",
                                            cls=(
                                                "h-9 border rounded px-3 py-1.5 w-full border-slate-300 bg-white text-slate-900 "
                                                "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                                            ),
                                        ),
                                        cls="flex-1"
                                    ),
                                    cls="flex gap-3 mb-3"
                                ),
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
                                style="display:none",
                                cls="border-t pt-3 mt-2"
                            ),
                            cls="flex flex-col"
                        ),
                        Div(
                            Button(
                                "Fetch new papers",
                                type="submit",
                                formaction="/fetch",
                                formmethod="post",
                                id="btn_fetch",
                                onclick=(
                                    # Change label + style immediately; disable after a tick so submit proceeds
                                    "this.textContent='Fetching…'; this.classList.add('opacity-50','cursor-not-allowed');"
                                    "setTimeout(()=>{ this.disabled=true; }, 10);"
                                ),
                                cls="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700",
                            ),
                            cls="mt-2"
                        ),
                        cls="space-y-4"
                    ),
                    None,
                    cls="bg-white dark:bg-slate-800 dark:border-slate-700 p-4 rounded-lg border"
                ),
                cls="container mx-auto max-w-3xl p-4"
            ),
            cls="mx-auto"
        ),
        cls="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 m-0 p-0"
    )
)


@rt("/fetch")
async def fetch(category: str, interest: str = "", summary_style: str = "", use_embeddings: str = "on", top_k: str = "10", verbosity: str = "medium", reasoning: str = "medium"):
    state = _load_state()
    # Use session anchor (stable during app lifetime)
    since = _get_session_anchor(category)

    # If using embeddings, fetch without simple substring filter to avoid over-pruning
    items = await arxiv_fetch(category, since, None if (use_embeddings != "off" and interest) else (interest or None))

    # Save user prefs for next time (but do NOT advance anchor here)
    prefs = _get_prefs(state)
    prefs.update(
        {
            "category": category,
            "interest": interest,
            "use_embeddings": ("off" if use_embeddings == "off" else "on"),
            "top_k": int(top_k or "10"),
            "summary_style": summary_style,
            "verbosity": verbosity,
            "reasoning": reasoning,
        }
    )
    _set_prefs(state, prefs)
    # Update history for quick re-use
    history = _get_history(state)
    if interest.strip():
        _push_history(history, "interests", interest, category=category)
    if summary_style.strip():
        _push_history(history, "summary_styles", summary_style)
    _set_history(state, history)
    _save_state(state)

    # Optional narrowing via Chroma embeddings
    narrowed_items = items
    narrowing_error = None
    try:
        if (use_embeddings != "off") and interest:
            k = max(5, min(200, int(top_k or "10")))
            # Offload Chroma ops to a thread to keep loop responsive
            narrowed_items = await asyncio.to_thread(narrow_with_chroma, items, interest, k, category)
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

    # Track latest seen and session touch for shutdown anchor advance
    _update_latest_seen(category, narrowed_items)

    results = [paper_row(it) for it in narrowed_items]
    fallback_ui = None
    if not results:
        # Automatic fallback: show recent cached matches (last 7 days) with current filters
        try:
            k = max(5, min(200, int(top_k or "10")))
        except Exception:
            k = 50
        fallback_days = 7
        cutoff = datetime.now(timezone.utc) - timedelta(days=fallback_days)

        try:
            client = _chroma_client()
            cache = client.get_or_create_collection(name="arxiv_cache")
            where: Optional[Dict[str, Any]] = ({"category": category} if category else None)
            if not interest.strip() or use_embeddings == "off":
                q = await asyncio.to_thread(cache.get, where=where)
                ids = q.get("ids") or []
                mds = q.get("metadatas") or []
                docs = q.get("documents") or []
                if ids and isinstance(ids[0], list):
                    flat_ids = []
                    for sub in ids:
                        flat_ids.extend(sub)
                    ids = flat_ids
                    mds = [x for sub in (mds or []) for x in (sub or [])]
                    docs = [x for sub in (docs or []) for x in (sub or [])]
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
                    if pub and pub < cutoff:
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
                items_all.sort(key=lambda x: x["published"] or cutoff, reverse=True)
                prev_items = items_all[:k]
            else:
                want = min(max(k * 3, k), 300)
                res = await asyncio.to_thread(cache.query, query_texts=[interest], n_results=want, where=where)
                ids = (res.get("ids") or [[]])[0]
                mds = (res.get("metadatas") or [[]])[0]
                docs = (res.get("documents") or [[]])[0]
                prev_items: List[Dict[str, Any]] = []
                cutoff_ts = cutoff.timestamp()
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
                    prev_items.append(
                        {
                            "id": f"https://arxiv.org/abs/{i}",
                            "arxiv_id": i,
                            "title": md.get("title", ""),
                            "authors": md.get("authors", ""),
                            "published": pub or cutoff,
                            "summary": doc or "",
                        }
                    )
                prev_items = prev_items[:k]

            prev_ui = [paper_row(it) for it in prev_items]
            if not prev_ui:
                prev_ui = [Div("No cached matches in the last 7 days.", cls="p-3 text-slate-600 dark:text-slate-300")]
            fallback_ui = Div(
                Div(
                    P(
                        f"No new papers since your last checked time ({_human(since)}).",
                        cls="text-sm text-slate-700 dark:text-slate-300",
                    ),
                    P(
                        "Showing recent cached matches instead (last 7 days).",
                        cls="text-xs text-slate-500 dark:text-slate-400 mb-2",
                    ),
                ),
                Div(*prev_ui, cls="grid grid-cols-1 gap-4"),
                cls="mt-4 p-3 rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40",
            )
        except Exception as _e:
            results = [Div("No new papers found for the chosen filters.", cls="p-4 text-slate-600 dark:text-slate-300")]  # type: ignore[assignment]

    return Html(
        Head(Title("ArXiv Results"), tailwind, htmx, marked_js, dompurify_js, markdown_script),
        Body(
            Main(
                Div(
                    H1("Results", cls="text-2xl font-bold mb-4"),
                Form(
                    Div(
                        Input(type="hidden", name="category", value=category),
                        Input(type="hidden", name="interest", value=interest),
                        Input(type="hidden", name="use_embeddings", value=("on" if use_embeddings != "off" else "off")),
                        Input(type="hidden", name="top_k", value=str(top_k)),
                        Input(type="hidden", name="verbosity", value=verbosity),
                        Input(type="hidden", name="reasoning", value=reasoning),
                        Textarea(
                            summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
                            name="summary_style",
                            cls="hidden",
                        ),
                    ),
                    Div(*results, cls="grid grid-cols-1 gap-4"),
                    (fallback_ui if fallback_ui is not None else None),
                    Div(
                        Button(
                            "Download & Summarize",
                            type="submit",
                            formaction="/download",
                            formmethod="post",
                            id="btn_download",
                            disabled=True,
                            onclick=(
                                "if(this.disabled){event.preventDefault(); return false;}"
                                "this.textContent='Downloading…'; this.classList.add('opacity-50','cursor-not-allowed');"
                                "setTimeout(()=>{ this.disabled=true; }, 10);"
                            ),
                            cls="inline-flex items-center justify-center h-10 px-4 bg-emerald-600 text-white rounded hover:bg-emerald-700 font-medium text-sm opacity-50 cursor-not-allowed",
                        ),
                        A(
                            "Back",
                            href="/",
                            cls="block h-10 px-4 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded no-underline font-medium text-sm text-center leading-10 hover:bg-slate-300 dark:hover:bg-slate-600",
                        ),
                        Input(
                            type="number",
                            name="previous_days",
                            value="7",
                            min="1",
                            max="60",
                            cls="w-20 h-10 border rounded px-3 border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-center",
                        ),
                        Button(
                            "Show previous matches",
                            type="submit",
                            formaction="/previous",
                            formmethod="post",
                            cls="inline-flex items-center justify-center h-10 px-4 bg-slate-600 text-white rounded hover:bg-slate-700 font-medium text-sm",
                        ),
                        cls="grid grid-cols-2 gap-3 items-center mt-4",
                    ),
                    cls="",
                ),
                Div(
                    P(
                        f"Last checked: {_human(since)} | fetched={len(items)} | semantic={'on' if use_embeddings!='off' else 'off'} | top_k={top_k} | narrowed={len(narrowed_items)} | interest='{interest}'",
                        cls="text-xs text-slate-500 dark:text-slate-400 mt-2",
                    ),
                    P(f"Chroma error: {narrowing_error}", cls="text-xs text-red-500") if narrowing_error else None,
                ),
                cls="container mx-auto max-w-3xl p-4 bg-white dark:bg-slate-800 dark:border-slate-700 rounded-lg border",
            ),
            cls="mx-auto"
        ),
        cls="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 m-0 p-0"
    )
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
async def download(request: Request, category: str = "", interest: str = "", summary_style: str = "", verbosity: str = "medium", reasoning: str = "medium"):
    form = await request.form()
    # Multiple checkbox values under key 'selected'
    selected_ids = form.getlist("selected")  # type: ignore[attr-defined]
    # Preserve settings to reconstruct results
    use_embeddings = form.get("use_embeddings") or "on"  # type: ignore[attr-defined]
    top_k_val = form.get("top_k") or "10"  # type: ignore[attr-defined]
    if not selected_ids:
        return Html(
            Head(Title("No Selection"), tailwind, htmx, marked_js, dompurify_js, markdown_script),
            Body(
                Main(
                    Div(
                        P("No papers selected."),
                        A(
                            "Back",
                            href="#",
                            onclick="history.back(); return false;",
                            cls="inline-block mt-2 px-3 py-2 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded",
                        ),
                        cls="container mx-auto max-w-3xl p-4"
                    ),
                    cls="mx-auto"
                ),
                cls="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 m-0 p-0"
            )
        )

    results_ui: List[Any] = []
    meta_client = arxiv.Client()
    for arxid in selected_ids:
        cache_dir = _paper_cache_dir(arxid)
        pdf_path = cache_dir / "original.pdf"
        txt_path = cache_dir / "text.txt"
        sum_path = cache_dir / "summary.txt"

        try:
            # Ensure PDF exists in cache
            if not pdf_path.exists():
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
            # Ensure extracted text exists
            if not txt_path.exists():
                text = await asyncio.to_thread(_pdf_text, pdf_path)
                txt_path.write_text(text)
            else:
                text = txt_path.read_text()
            # Ensure summary exists; only generate if missing
            if not sum_path.exists():
                summary = await asyncio.to_thread(
                    openai_summarize,
                    text,
                    summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
                    verbosity,
                    reasoning,
                    title,
                )
                sum_path.write_text(summary)
            else:
                summary = sum_path.read_text()
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

            sum_id = f"sum-{_css_id(arxid)}"
            regen_btn = Button(
                "Regenerate summary",
                type="button",
                onclick=(
                    "console.log('[DEBUG] Regenerate clicked for', this.dataset.arxivId);"
                    "console.log('[DEBUG] htmx available?', typeof window.htmx !== 'undefined');"
                    "console.log('[DEBUG] Button attributes:', this.attributes);"
                    "this.textContent='Regenerating…'; this.classList.add('opacity-50','cursor-not-allowed');"
                    "setTimeout(()=>{ this.disabled=true; }, 10);"
                    # Fallback if htmx is unavailable
                    "if(!window.htmx){"
                        "console.log('[DEBUG] Using fetch fallback');"
                        "const tgt=this.dataset.target; const id=this.dataset.arxivId; const style=this.dataset.summaryStyle||''; const verb=this.dataset.verbosity||'medium'; const reas=this.dataset.reasoning||'medium';"
                        "fetch('/regenerate', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:new URLSearchParams({arxiv_id:id, summary_style:style, verbosity:verb, reasoning:reas})})"
                        ".then(r=>r.text()).then(html=>{ const el=document.querySelector(tgt); if(el){ el.outerHTML=html; if(window.__renderMarkdown){ try{ window.__renderMarkdown(document.querySelector(tgt)); }catch(_e){} } } })"
                        ".catch(err=>{ console.error('[DEBUG] Fetch error:', err); this.textContent='Regenerate summary'; this.disabled=false; this.classList.remove('opacity-50','cursor-not-allowed'); });"
                    "} else {"
                        "console.log('[DEBUG] htmx should handle this click');"
                    "}"
                ),
                cls="mt-3 inline-flex items-center justify-center h-9 px-3 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded hover:bg-slate-300 dark:hover:bg-slate-600 text-sm",
                **{
                    "hx-post": "/regenerate",
                    "hx-target": f"#{sum_id}",
                    "hx-swap": "outerHTML",
                    "hx-trigger": "click",
                    "hx-vals": json.dumps({
                        "arxiv_id": arxid,
                        "summary_style": summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
                        "verbosity": verbosity,
                        "reasoning": reasoning,
                    }),
                    "hx-on--before-request": "console.log('[DEBUG] htmx before-request event fired')",
                    "hx-on--after-request": "console.log('[DEBUG] htmx after-request event fired')",
                    "hx-on--config-request": "console.log('[DEBUG] htmx config-request event fired', event.detail)",
                },
                **{
                    "data-arxiv-id": arxid,
                    "data-summary-style": (summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding"),
                    "data-verbosity": verbosity,
                    "data-reasoning": reasoning,
                    "data-target": f"#{sum_id}",
                },
            )

            results_ui.append(
                Div(
                    Div(
                        H3(title, cls="text-xl font-semibold text-slate-900 dark:text-slate-100"),
                        P(f"Authors: {authors}", cls="text-sm text-slate-600 dark:text-slate-300") if authors else None,
                        P(f"Published: {published_str}", cls="text-xs text-slate-500 dark:text-slate-400") if published_str else None,
                        cls="mb-2"
                    ),
                    Div(
                        Div(
                            summary,
                            cls="mt-2 leading-relaxed text-[0.95rem]",
                            **{"data-md": "1"}
                        ),
                        Div(cls="mt-4 border-t border-slate-200 dark:border-slate-500"),
                        Div(
                            A(
                                "Open PDF",
                                href=(f"/files/pdf/{pdf_uid}" if pdf_uid else f"https://arxiv.org/pdf/{arxid}.pdf"),
                                target="_blank",
                                cls="inline-flex items-center justify-center h-9 px-3 bg-indigo-600 text-white rounded hover:bg-indigo-700 text-sm",
                            ),
                            regen_btn,
                            cls="mt-3 flex justify-between items-center gap-3"
                        ),
                        id=sum_id,
                    ),
                    cls="p-5 border rounded-xl shadow-sm bg-white dark:bg-slate-800 dark:border-slate-700"
                )
            )
        except Exception as e:
            results_ui.append(Div(H3(f"{arxid}"), P(f"Error: {e}", cls="text-red-600"), cls="p-3 border rounded"))

    return Html(
        Head(Title("Download + Summaries"), tailwind, htmx, marked_js, dompurify_js, markdown_script),
        Body(
            Main(
                Div(
                    P("Using per-paper cache under papers/<arXiv ID>", cls="text-sm text-slate-600 dark:text-slate-300"),
                Div(*results_ui, cls="mt-4 space-y-3"),
                Div(
                    Form(
                    Input(type="hidden", name="category", value=category),
                    Input(type="hidden", name="interest", value=interest),
                    Input(type="hidden", name="use_embeddings", value=("on" if (use_embeddings != "off") else "off")),
                    Input(type="hidden", name="top_k", value=str(top_k_val)),
                    Input(type="hidden", name="verbosity", value=verbosity),
                    Input(type="hidden", name="reasoning", value=reasoning),
                    Textarea(summary_style or "", name="summary_style", cls="hidden"),
                    Button(
                        "Back to results",
                        type="submit",
                        formaction="/fetch",
                        formmethod="post",
                        cls="inline-flex items-center justify-center h-10 px-4 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded hover:bg-slate-300 dark:hover:bg-slate-600 font-medium text-sm mt-4",
                    ),
                    ),
                ),
                cls="container mx-auto max-w-3xl p-4"
            ),
            cls="mx-auto"
        ),
        cls="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 m-0 p-0"
    )
)


@rt("/previous")
async def previous(category: str, interest: str = "", use_embeddings: str = "on", top_k: str = "10", previous_days: str = "7", summary_style: str = "", verbosity: str = "medium", reasoning: str = "medium"):
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
        # Offload Chroma get to a thread
        q = await asyncio.to_thread(cache.get, where=where)
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
                    pub = dateparser.parse(md.get("published", ""))
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
        want = min(max(k * 3, k), 300)
        # Offload Chroma query to a thread
        res = await asyncio.to_thread(cache.query, query_texts=[interest], n_results=want, where=where)
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

    return Html(
        Head(Title("arXiv Helper"), tailwind, htmx, marked_js, dompurify_js, markdown_script),
        Body(
            Main(
                Div(
                    H1("Previous Matches", cls="text-2xl font-bold mb-4"),
                Form(
                    Div(
                        Input(type="hidden", name="category", value=category),
                        Input(type="hidden", name="interest", value=interest),
                        Input(type="hidden", name="use_embeddings", value=("on" if use_embeddings != "off" else "off")),
                        Input(type="hidden", name="top_k", value=str(k)),
                        Input(type="hidden", name="previous_days", value=str(days)),
                        Input(type="hidden", name="verbosity", value=verbosity),
                        Input(type="hidden", name="reasoning", value=reasoning),
                        Textarea(summary_style or "", name="summary_style", cls="hidden"),
                    ),
                    Div(*results, cls="grid grid-cols-1 gap-4"),
                    Div(
                        Button(
                            "Download & Summarize",
                            type="submit",
                            formaction="/download",
                            formmethod="post",
                            id="btn_download",
                            disabled=True,
                            onclick=(
                                "if(this.disabled){event.preventDefault(); return false;}"
                                "this.textContent='Downloading…'; this.classList.add('opacity-50','cursor-not-allowed');"
                                "setTimeout(()=>{ this.disabled=true; }, 10);"
                            ),
                            cls="inline-flex items-center justify-center h-10 px-4 bg-emerald-600 text-white rounded hover:bg-emerald-700 font-medium text-sm opacity-50 cursor-not-allowed",
                        ),
                        A(
                            "Back",
                            href="#",
                            onclick="history.back(); return false;",
                            cls="block h-10 px-4 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded no-underline font-medium text-sm text-center leading-10",
                        ),
                        cls="grid grid-cols-2 gap-3 items-center mt-4"
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
            cls="mx-auto"
        ),
        cls="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 m-0 p-0"
    )
)


# Removed legacy generic /files route in favor of UUID-based routes above


# ---------- Manage Categories UI ----------


@rt("/categories")
def categories_page(error: str | None = None, saved: str | None = None, q: str | None = None):
    state = _load_state()
    cats = _get_user_categories(state)
    lines = "\n".join(f"{it['label']}|{it['code']}" for it in cats)
    flash = None
    if error:
        flash = Div(error, cls="mb-3 p-2 rounded bg-red-50 text-red-700 border border-red-200")
    elif saved:
        flash = Div("Saved categories.", cls="mb-3 p-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-200")
    # Suggestions via local cache and fuzzy search
    suggestions_ui = None
    cache = _load_category_cache()
    if q and cache:
        sugg = _fuzzy_search_categories(q, cache, limit=10)
        if sugg:
            rows = []
            for it in sugg:
                label = it.get("label", "")
                code = it.get("code", "")
                rows.append(
                    Label(
                        Input(
                            type="checkbox",
                            name="sugg",
                            value=code,
                            **{"data-label": label},
                            cls=(
                                "h-4 w-4 shrink-0 rounded-sm border-2 border-slate-400 dark:border-slate-500 "
                                "bg-white dark:bg-slate-900 accent-emerald-600 focus:ring-2 focus:ring-emerald-500"
                            ),
                        ),
                        Span(f"{label} ({code})", cls="ml-2 text-sm"),
                        cls=(
                            "flex items-center py-1 px-2 rounded-md cursor-pointer select-none transition-colors "
                            "hover:bg-slate-200 dark:hover:bg-slate-700 hover:ring-1 hover:ring-slate-300 "
                            "dark:hover:ring-slate-600"
                        )
                    )
                )
            suggestions_ui = Div(
                H3("Suggestions", cls="font-medium text-sm mb-2"),
                Div(*rows, id="sugg_box", cls="divide-y divide-slate-200 dark:divide-slate-700 border rounded p-2 bg-slate-50 dark:bg-slate-800/40"),
                P("Selected suggestions will be added when you Save.", cls="text-xs text-slate-500 mt-2"),
                cls="mt-2"
            )
        else:
            suggestions_ui = P("No matches.", cls="text-xs text-slate-500 mt-2")
    elif q and not cache:
        suggestions_ui = P("Category cache not available. Run the fetch script to enable search.", cls="text-xs text-slate-500 mt-2")
    return Html(
        Head(Title("Manage Categories"), tailwind, htmx, marked_js, dompurify_js, markdown_script),
        Body(
            Main(
                Div(
                H1("Manage Categories", cls="text-2xl font-bold mb-3"),
                P("Edit your category list below. One per line as 'Label|code' (e.g., 'Computation and Language|cs.CL').", cls="text-sm text-slate-600 dark:text-slate-300 mb-3"),
                flash,
                Form(
                    Div(
                        Label("Search catalog (local cache)", cls="text-sm text-slate-600 dark:text-slate-300"),
                        Div(
                            Input(
                                name="q",
                                value=q or "",
                                placeholder="e.g., language, robotics, cs.CL",
                                cls=(
                                    "flex-grow h-10 px-3 border rounded border-slate-300 bg-white text-slate-900 "
                                    "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700 "
                                    "placeholder:text-slate-400 dark:placeholder:text-slate-400"
                                ),
                            ),
                            Button(
                                "Search",
                                type="submit",
                                formaction="/categories",
                                formmethod="get",
                                cls=(
                                    "flex-1 h-10 px-3 ml-2 bg-slate-200 hover:bg-slate-300 "
                                    "dark:bg-slate-700 dark:hover:bg-slate-600 dark:text-slate-100 "
                                    "rounded text-sm"
                                ),
                            ),
                            A(
                                "Clear",
                                href="/categories",
                                cls="h-10 px-3 ml-2 rounded text-sm text-slate-600 dark:text-slate-300",
                            ),
                            cls="flex items-center"
                        ),
                        None,
                        cls="mb-4"
                    ),
                ),
                Form(
                    (suggestions_ui if suggestions_ui else None),
                    Textarea(
                        lines,
                        name="bulk",
                        id="cats_bulk",
                        rows=14,
                        cls=(
                            "w-full border rounded p-2 border-slate-300 bg-white text-slate-900 "
                            "dark:bg-slate-900 dark:text-slate-100 dark:border-slate-700"
                        ),
                    ),
                    Div(
                        Button(
                            "Save",
                            type="submit",
                            formaction="/categories/save",
                            formmethod="post",
                            cls="h-10 px-4 bg-emerald-600 text-white rounded hover:bg-emerald-700 font-medium text-sm",
                        ),
                        Button(
                            "Reset to defaults",
                            type="submit",
                            formaction="/categories/reset",
                            formmethod="post",
                            cls="h-10 px-4 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded font-medium text-sm",
                        ),
                        A(
                            "Back",
                            href="/",
                            cls="h-10 px-4 bg-slate-100 dark:bg-slate-800 dark:text-slate-100 border border-slate-300 dark:border-slate-700 rounded inline-flex items-center justify-center no-underline text-sm",
                        ),
                        cls="mt-3 flex gap-3"
                    ),
                ),
                cls="container mx-auto max-w-3xl p-4 bg-white dark:bg-slate-800 dark:border-slate-700 rounded-lg border"
            ),
            cls="mx-auto"
        ),
        cls="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 m-0 p-0"
    )
)


def _parse_categories_bulk(bulk: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for raw in (bulk or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # allow either "Label|code" or "code" alone
        if "|" in line:
            label, code = line.split("|", 1)
            label = label.strip()
            code = code.strip()
        else:
            code = line
            label = ""
        if not code:
            continue
        out.append({"label": label, "code": code})
    return out


@rt("/categories/save")
def categories_save(bulk: str = "", sugg: list[str] = []):
    items = _parse_categories_bulk(bulk)
    # Merge any selected suggestions by code, looking up labels from cache
    if sugg:
        selected = [str(x) for x in sugg]
        cache = _load_category_cache()
        code_to_label = {it.get("code", ""): it.get("label", "") for it in cache}
        for code in selected:
            code = (code or "").strip()
            if not code:
                continue
            label = code_to_label.get(code, "") or code
            items.append({"label": label, "code": code})
    if not items:
        return categories_page(error="Please provide at least one category entry.")
    state = _load_state()
    _set_user_categories(state, items)
    _save_state(state)
    return categories_page(saved="1")


@rt("/categories/reset")
def categories_reset():
    state = _load_state()
    defaults = [{"label": name.split(" (", 1)[0], "code": code} for name, code in CATEGORIES.items()]
    _set_user_categories(state, defaults)
    _save_state(state)
    return categories_page(saved="1")


@app.on_event("startup")
async def _maybe_fresh_state() -> None:  # type: ignore[attr-defined]
    if os.getenv("FRESH") == "1":
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("[DEBUG] FRESH=1: removed state.json", flush=True)
        except Exception as e:
            print(f"[DEBUG] FRESH=1: could not remove state.json: {e}", flush=True)
    # Initialize session anchors lazily when categories are used; nothing to do here otherwise


@app.on_event("shutdown")
async def _persist_session_anchors() -> None:  # type: ignore[attr-defined]
    # Advance per-category "last checked" when app closes
    if not TOUCHED_CATEGORIES:
        return
    state = _load_state()
    for cat in list(TOUCHED_CATEGORIES):
        # Prefer the latest seen published time; fallback to now if none
        dt = LATEST_SEEN.get(cat) or datetime.now(timezone.utc)
        _set_last_run(state, cat, dt.isoformat())
    _save_state(state)
    print("[DEBUG] shutdown: advanced last checked for categories:", sorted(TOUCHED_CATEGORIES), flush=True)


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


@rt("/regenerate")
async def regenerate(
    arxiv_id: str,
    request: Request | None = None,
    summary_style: str = "",
    verbosity: str = "medium",
    reasoning: str = "medium",
    htmx: HtmxHeaders | None = None,
):
    # Overwrite the cached summary for a single paper using the current style
    has_hx_header = False
    try:
        if request is not None:
            hx = request.headers.get("hx-request") or request.headers.get("HX-Request")
            has_hx_header = (hx or "").lower() == "true"
    except Exception:
        has_hx_header = False
    is_htmx = has_hx_header or (htmx is not None)
    print(
        f"[DEBUG] /regenerate called arxiv_id={arxiv_id} hx_header={has_hx_header} htmx_obj={htmx is not None}",
        flush=True,
    )
    cache_dir = _paper_cache_dir(arxiv_id)
    pdf_path = cache_dir / "original.pdf"
    txt_path = cache_dir / "text.txt"
    sum_path = cache_dir / "summary.txt"

    # Ensure prerequisites
    if not pdf_path.exists():
        await asyncio.to_thread(_download_pdf_via_arxiv, arxiv_id, pdf_path)
    if not txt_path.exists():
        text = await asyncio.to_thread(_pdf_text, pdf_path)
        txt_path.write_text(text)
    else:
        text = txt_path.read_text()

    # Fetch title for accurate heading handling
    title_val = None
    try:
        meta_client = arxiv.Client()
        r = next(meta_client.results(arxiv.Search(id_list=[arxiv_id])))
        title_val = (r.title or "").strip() or None
    except Exception:
        title_val = None

    # Regenerate unconditionally
    summary = await asyncio.to_thread(
        openai_summarize,
        text,
        summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
        verbosity,
        reasoning,
        title_val,
    )
    sum_path.write_text(summary)

    # If htmx request, return the updated summary block for in-place swap
    sum_id = f"sum-{_css_id(arxiv_id)}"
    pdf_uid = _register_file(pdf_path, "pdf", "application/pdf") if pdf_path.exists() else None
    block = Div(
        Div(
            summary,
            cls="mt-2 leading-relaxed text-[0.95rem]",
            **{"data-md": "1"}
        ),
        Div(cls="mt-4 border-t border-slate-200 dark:border-slate-500"),
        Div(
            A(
                "Open PDF",
                href=(f"/files/pdf/{pdf_uid}" if pdf_uid else f"https://arxiv.org/pdf/{arxiv_id}.pdf"),
                target="_blank",
                cls="inline-flex items-center justify-center h-9 px-3 bg-indigo-600 text-white rounded hover:bg-indigo-700 text-sm",
            ),
            Button(
                "Regenerate summary",
                type="button",
                onclick=(
                    "console.log('[DEBUG] Regenerate clicked in regenerate route for', this.dataset.arxivId);"
                    "console.log('[DEBUG] htmx available?', typeof window.htmx !== 'undefined');"
                    "this.textContent='Regenerating…'; this.classList.add('opacity-50','cursor-not-allowed');"
                    "setTimeout(()=>{ this.disabled=true; }, 10);"
                    "if(!window.htmx){"
                        "console.log('[DEBUG] Using fetch fallback in regenerate route');"
                        "const tgt=this.dataset.target; const id=this.dataset.arxivId; const style=this.dataset.summaryStyle||''; const verb=this.dataset.verbosity||'medium'; const reas=this.dataset.reasoning||'medium';"
                        "fetch('/regenerate', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:new URLSearchParams({arxiv_id:id, summary_style:style, verbosity:verb, reasoning:reas})})"
                        ".then(r=>r.text()).then(html=>{ const el=document.querySelector(tgt); if(el){ el.outerHTML=html; if(window.__renderMarkdown){ try{ window.__renderMarkdown(document.querySelector(tgt)); }catch(_e){} } } })"
                        ".catch(err=>{ console.error('[DEBUG] Fetch error:', err); this.textContent='Regenerate summary'; this.disabled=false; this.classList.remove('opacity-50','cursor-not-allowed'); });"
                    "} else {"
                        "console.log('[DEBUG] htmx should handle this click in regenerate route');"
                    "}"
                ),
                cls="inline-flex items-center justify-center h-9 px-3 bg-slate-200 dark:bg-slate-700 dark:text-slate-100 rounded hover:bg-slate-300 dark:hover:bg-slate-600 text-sm",
            **{
                "hx-post": "/regenerate",
                "hx-target": f"#{sum_id}",
                "hx-swap": "outerHTML",
                "hx-trigger": "click",
                "hx-vals": json.dumps({
                    "arxiv_id": arxiv_id,
                    "summary_style": summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding",
                    "verbosity": verbosity,
                    "reasoning": reasoning,
                }),
                "hx-on--before-request": "console.log('[DEBUG] htmx before-request event fired in regenerate route')",
                "hx-on--after-request": "console.log('[DEBUG] htmx after-request event fired in regenerate route')",
                "hx-on--config-request": "console.log('[DEBUG] htmx config-request event fired in regenerate route', event.detail)",
            },
            **{
                "data-arxiv-id": arxiv_id,
                "data-summary-style": (summary_style or "Someone with passing knowledge of the area, but not an expert - use clear, understandable terms that don't need deep specialist understanding"),
                "data-verbosity": verbosity,
                "data-reasoning": reasoning,
                "data-target": f"#{sum_id}",
            },
        ),
            cls="mt-3 flex justify-between items-center gap-3"
        ),
        # Ensure markdown render runs even if htmx events are missed
        Script(f"try{{ window.__renderMarkdown && window.__renderMarkdown(document.getElementById('{sum_id}')); }}catch(_e){{}}"),
        id=sum_id,
    )
    return block
