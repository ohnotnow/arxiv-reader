# History Feature Implementation Checklist

- [x] Define history schema in code (no migration needed; treat missing as empty)
- [x] Add `_get_history` and `_set_history` helpers
- [x] Add normalization helpers: `_normalize_interest`, `_normalize_style`
- [x] Add `_push_history(history, kind, value, category=None, cap=25)` utility
- [x] Update `/fetch` handler to push non-empty `interest` to `history.interests`
- [x] Update `/fetch` handler to push non-empty `summary_style` to `history.summary_styles`
- [x] Persist updated `history` in `state.json` alongside `prefs` and `last_run`
- [x] Load history in `index` route
- [x] Build recent interests list (category-first, then global) and de-duplicate
- [x] Add `datalist#recent-interests` and link to the interest input
- [x] Add a visible "Recent interests" select that copies into the input
- [ ] Optional: add small interest “chips” (last ~5) that set the input via `data-value`
- [x] Build recent styles list (most recent first) and de-duplicate
- [x] Add a “Recent styles” `select` that copies the chosen value into the textarea on change
- [ ] Optional: add style “chips” as quick-fill buttons
- [x] Truncate long labels in select/chips (e.g., 80 chars) while using full value
- [x] Ensure all user text is safely escaped; avoid embedding raw strings into JS literals
- [x] Cap list sizes at 25 items on insert
- [ ] Quick manual QA across two categories to verify ordering and reuse
- [ ] Document behavior in README (short note under usage)

---

# Session-Based "Last checked" Flow

- [x] Add session-scoped anchors per category (in-memory)
- [x] Use session anchor for `since` in `/fetch` (don’t advance immediately)
- [x] Track `latest_seen` per category during the session
- [x] On `shutdown`, persist `last_run` per touched category to `latest_seen` (fallback: now)
- [x] Update UI copy to show “Last checked … Updates when you close the app”
- [x] In `/fetch`, if no new items, auto-render recent cached matches (last 7 days)
- [ ] Manual QA: repeated fetches with filter tweaks yield consistent window
- [ ] Manual QA: close app → reopen → “Last checked” advanced
- [ ] Optional: add small banner on results clarifying fallback was used

## UI polish

- [x] Show per-category “Last checked” inside category dropdown options
- [x] Update “Last checked for …” text live when category selection changes
- [x] Rename button to “Fetch new papers” and remove bottom “Last checked / updates when you close the app” copy

## Category cache

- [x] Add script `scripts/fetch_categories.py` to scrape taxonomy and write `data/arxiv_categories.json`
- [x] Add a seed `data/arxiv_categories.json` with common categories
- [x] Document usage in README
- [ ] Optional: validate codes via a tiny query (future)
- [ ] Optional: add fuzzy search helper using local cache (future)
  - [x] Implemented basic fuzzy search (token overlap + substring) in Manage Categories

## Manage categories UI

- [x] Load user-curated categories from `state["categories"]` with fallback to defaults
- [x] Render dropdown from curated list (ordered), with per-category "Last checked"
- [x] Add "Manage categories" page with bulk edit textarea (Label|code per line)
- [x] Save/update curated list in state; reset to defaults
- [ ] Optional: add drag-reorder UI and field-by-field editing
- [ ] Optional: confirm prompt on removing a category
