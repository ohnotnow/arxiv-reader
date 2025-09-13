#!/usr/bin/env python3
"""
Fetch and cache arXiv subject categories from the taxonomy page.

Outputs JSON to data/arxiv_categories.json by default:
[
  {"code": "cs.CL", "label": "Computation and Language", "group": "cs", "updated_at": "...", "source": "arxiv_category_taxonomy"},
  ...
]

Notes:
- There is no official arXiv API endpoint to list all subject classes; this scrapes the public taxonomy HTML.
- Parser targets <h4>cs.CL <span>(Computation and Language)</span></h4> patterns.
- The "group" is derived from the code prefix before the first dot (e.g., "cs").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import urlopen, Request


TAXONOMY_URL = "https://arxiv.org/category_taxonomy"


class TaxonomyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_h4 = False
        self.in_span = False
        self.current_code: Optional[str] = None
        self.current_label_parts: List[str] = []
        self.items: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() == "h4":
            self.in_h4 = True
            self.in_span = False
            self.current_code = None
            self.current_label_parts = []
        elif self.in_h4 and tag.lower() == "span":
            self.in_span = True

    def handle_endtag(self, tag: str):
        if tag.lower() == "h4" and self.in_h4:
            # finalize
            label_raw = "".join(self.current_label_parts).strip()
            # labels often appear like "(Computation and Language)"
            label = label_raw.strip()
            if label.startswith("(") and label.endswith(")"):
                label = label[1:-1].strip()
            if self.current_code and label:
                group = self.current_code.split(".", 1)[0]
                self.items.append({
                    "code": self.current_code,
                    "label": label,
                    "group": group,
                })
            # reset
            self.in_h4 = False
            self.in_span = False
            self.current_code = None
            self.current_label_parts = []
        elif tag.lower() == "span" and self.in_h4:
            self.in_span = False

    def handle_data(self, data: str):
        if not self.in_h4:
            return
        if self.in_span:
            self.current_label_parts.append(data)
        else:
            # Look for a code token like cs.CL, math.AG, stat.ML, eess.AS, etc.
            text = data.strip()
            m = re.search(r"\b([a-z]{2,}\.[A-Za-z0-9\-]+)\b", text)
            if m and not self.current_code:
                self.current_code = m.group(1)


def fetch_taxonomy(url: str) -> str:
    req = Request(url, headers={"User-Agent": "arxiv-helper/0.1 (+https://arxiv.org)"})
    with urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def parse_taxonomy(html: str) -> List[Dict[str, str]]:
    parser = TaxonomyParser()
    parser.feed(html)
    # Deduplicate by code (first wins), keep sorted by code
    seen = set()
    out: List[Dict[str, str]] = []
    for it in parser.items:
        code = it.get("code")
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(it)
    out.sort(key=lambda x: x.get("code", ""))
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fetch arXiv categories into a local JSON cache")
    ap.add_argument("--url", default=TAXONOMY_URL, help="Taxonomy URL")
    ap.add_argument("--out", default="data/arxiv_categories.json", help="Output JSON path")
    ap.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing file")
    args = ap.parse_args(argv)

    html = fetch_taxonomy(args.url)
    items = parse_taxonomy(html)
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = [
        {
            **it,
            "updated_at": now_iso,
            "source": "arxiv_category_taxonomy",
        }
        for it in items
    ]

    if args.stdout:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    print(f"Wrote {len(payload)} categories to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

