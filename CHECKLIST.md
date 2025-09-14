# Named Summary Styles — Implementation Plan

Goal: Replace the free‑text, recents‑only summary style UX with a simple named styles system that lets users pick, edit, and delete styles while still passing only the style body to the LLM.

Core idea
- Saved styles live in `state.json` under `styles.items: [{ title, body, ts, count }]`.
- The UI shows a dropdown of style titles, a delete button, a title input, and a body textarea.
- Selecting a style fills title + textarea. The textarea (`name=summary_style`) remains the value sent to OpenAI.
- Save/Update overwrites an existing title (case‑insensitive) or creates a new one.
- Delete removes the selected style and refreshes the UI.

Notes
- Keep it minimal: no pinning/favorites. A handful of styles is expected.
- No migration required for external users; optional one‑time lift from current recents is listed at the end.

---

## Implementation Checklist

Data model & helpers
- [ ] Add `styles` to `state.json` schema: `{ "items": [{ "title": str, "body": str, "ts": iso8601, "count": int }] }`.
- [ ] Implement `_get_styles(state) -> dict` and `_set_styles(state, styles: dict)` helpers.
- [ ] Add `normalize_title(title: str) -> str` (trim, collapse spaces, lower for uniqueness checks).
- [ ] Add helper to find item by title (case‑insensitive) and return index/object.
- [ ] Add `upsert_style(title: str, body: str)` (update if exists else add; set `ts` and bump `count`).
- [ ] Add `delete_style(title: str)` (remove by case‑insensitive match).

UI: index page styles section
- [ ] Replace “Recent styles” select with “Saved styles” select of titles (fallback to current recents if none saved).
- [ ] Add Delete button next to the select (disabled when no selection); confirm before delete.
- [ ] Add Title input (`id=style_title_input`), plain text.
- [ ] Keep Textarea as the style body (`name=summary_style`, `id=summary_style_input`).
- [ ] On select change: populate title input + textarea with the selected style’s data.
- [ ] Add Save/Update button that posts title+body; success re-renders only the styles section.

HTMX routes & partial rendering
- [ ] Add `@rt('/styles/save')` to validate title/body, upsert, persist state, and return the styles section partial.
- [ ] Add `@rt('/styles/delete')` to delete by title, persist, and return the styles section partial.
- [ ] Extract a `render_styles_section(state, current_title: str | None, current_body: str)` helper used by index/save/delete.
- [ ] Pass title/body map to the client safely (e.g., via `data-` attributes on `<option>` or inline JSON + `onchange` script).

Validation & UX
- [ ] Require non‑empty title and body; trim inputs.
- [ ] Enforce max title length (e.g., 60 chars); show inline error on save if exceeded.
- [ ] Uniqueness: titles compared case‑insensitively; saving with an existing title overwrites (explicit “Update” behavior).
- [ ] When deleting the currently selected style, clear title + textarea and selection.
- [ ] Keep the textarea as the single source of truth for the value sent to the LLM.

Behavior across routes
- [ ] Ensure `/fetch`, `/download`, `/previous`, and `/regenerate` continue to read `summary_style` from the textarea (no changes needed).

Styling polish
- [ ] Align controls (select, delete button, title input, textarea) within the existing layout.
- [ ] Add helpful placeholders: “Style title” and “Describe how to write the summary…”.
- [ ] Truncate long titles in the select with ellipsis; show full title on hover via `title` attribute.

Testing (manual)
- [ ] Create new style, save, select, edit, update, and delete.
- [ ] Confirm `state.json` updates correctly, including `ts` and `count`.
- [ ] Restart the app; verify saved styles persist and populate the dropdown.
- [ ] Run a summary and a regenerate with a saved style; confirm prompt body is used.
- [ ] Delete the selected style and verify the UI falls back gracefully.

Optional (later)
- [ ] One‑time lift from existing `history.summary_styles` into saved styles (generate title from first 50–60 chars).
- [ ] Export/Import styles as JSON.
- [ ] Preset templates (ELI5, Executive summary, Practitioner tips).

