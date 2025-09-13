# Feature Plan: History for Interests and Summary Styles

## Objective
Add lightweight “history” for specific interests and summary styles so users can quickly reuse past entries while keeping the existing freeform inputs unchanged.

## State Model
- Store history alongside existing `prefs` and `last_run` in `state.json`.
- New key: `history` with two lists.
  - `history.interests`: array of `{ value: str, ts: iso, category: str | "", count: int }`
  - `history.summary_styles`: array of `{ value: str, ts: iso, count: int }`

### Semantics
- De-duplicate by a normalized key when inserting:
  - Interests: trim, collapse whitespace, lowercase for dedupe key.
  - Styles: trim, collapse whitespace (case preserved in storage; case-insensitive or case-preserving dedupe is acceptable — prefer preserving original casing).
- On reuse: update `ts` to now and increment `count`.
- Order lists by most-recent-first.
- Cap each list to a fixed size (e.g., 25 items) to avoid unbounded growth.
- Backward compatible: if `history` missing, treat as empty.

### Example `state.json` shape
```
{
  "last_run": { "cs.AI": "2025-09-11T12:10:09.096846+00:00" },
  "prefs": {
    "category": "cs.AI",
    "interest": "llm persona",
    "use_embeddings": "on",
    "top_k": 10,
    "summary_style": "Someone with passing knowledge..."
  },
  "history": {
    "interests": [
      { "value": "retrieval-augmented generation", "ts": "2025-09-12T10:00:00Z", "category": "cs.CL", "count": 5 },
      { "value": "llm persona", "ts": "2025-09-11T10:00:00Z", "category": "cs.AI", "count": 2 }
    ],
    "summary_styles": [
      { "value": "For product engineers", "ts": "2025-09-12T10:00:00Z", "count": 3 },
      { "value": "For grad students", "ts": "2025-09-10T09:00:00Z", "count": 1 }
    ]
  }
}
```

## Write Points
- On `/fetch` (when the user clicks Fetch):
  - If `interest` is non-empty → push to `history.interests` with the current `category`.
  - If `summary_style` is non-empty → push to `history.summary_styles`.
- No changes required on `/download` — the selection was already made during fetch.

## UI Additions (keep freeform inputs)
1) Specific Interest (input[type=text])
- Keep the existing text input.
- Add a `datalist` populated from recent interests (most recent first):
  - Prefer items from the current `category` first, then fall back to global (other categories).
  - Input attribute: `list="recent-interests"`.
- Optional: Below the input, render a small chip row (last ~5) — clicking a chip sets the input value via a tiny inline script using `data-value` attributes.

2) Summary Style (textarea)
- Keep the existing textarea.
- Add a small `select` labeled “Recent styles” above/next to the textarea populated from `history.summary_styles` (most recent first).
- On `select` change, copy the selected option’s value into the textarea via a one-liner inline script.
- Optional: Provide chips similar to interests.

### Safety considerations
- Avoid directly interpolating user strings into JS literals. Use `data-value` attributes and read them in the handler, or set values via DOM properties.
- FastHTML components should handle HTML escaping for text nodes; keep user-provided strings out of raw HTML contexts.

## Helper Functions
- `_get_history(state) -> dict`: returns `state.get("history", {})`.
- `_set_history(state, history) -> None`.
- `_normalize_interest(value: str) -> str`: trim + collapse whitespace + lowercase.
- `_normalize_style(value: str) -> str`: trim + collapse whitespace (preserve case in stored `value`).
- `_push_history(history, kind, value, category=None, cap=25) -> None`:
  - Choose the normalize fn based on `kind`.
  - Search for existing by normalized key; if found, update `ts` and `count += 1` and move to front; else insert at front with `count=1`.
  - Trim to `cap` length.

## Index Route Data Preparation
- Load `history` and build:
  - `recent_interests`: unique by normalized value, category-filtered first, limited to ~10 for UI.
  - `recent_styles`: unique by normalized value, limited to ~10 for UI.
- Render the `datalist`/`select` and optional chips.

## Edge Cases
- Empty history: omit/disable extra UI; freeform inputs remain.
- Very long style text: truncate labels in UI (e.g., 80 chars) but keep full value.
- FRESH=1 removes `state.json`, which clears history — acceptable for dev.

## Testing Plan
- Manual QA:
  - Start app, enter several interests and styles, press Fetch, repeat across categories.
  - Verify `state.json` accumulates history with updated `ts`/`count` and caps lists.
  - Confirm interest datalist suggests recent items and category-prioritized ordering.
  - Confirm recent styles select populates the textarea correctly and safely.
- Optional unit tests (future):
  - Test normalization and `_push_history` behavior (dedupe, order, cap).

## Future Enhancements (nice-to-haves)
- Per-category history caps/toggling; clear-history buttons in UI.
- Pin/favorite specific entries to keep at top.
- Export/import of history via a small JSON file.

