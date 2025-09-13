# Regenerate Summary: Desired UX and Current State

## Goal
- Provide an in-place “Regenerate summary” action per paper on the summaries page.
- When clicked, the button should:
  - Show a loading state (text change + greyed out),
  - POST the current `arxiv_id` and `summary_style` to the server,
  - Receive the updated summary markup,
  - Replace only the paper’s summary block in the page (no full-page reload or redirect).

## What’s Implemented Now
- Per-paper cache directory: `papers/<arxiv_id>/original.pdf`, `text.txt`, `summary.txt`.
- Route `POST /regenerate`:
  - Ensures PDF/text exist,
  - Re-summarizes with the current `summary_style`,
  - Overwrites `summary.txt` and returns an updated summary block.
  - If the request is an htmx request, it returns just the summary block; otherwise it returns a minimal page.
- Summaries page markup:
  - Each summary block is wrapped in a container with a stable id `sum-<arxiv_id>`.
  - A “Regenerate summary” control intended to htmx-swap that container in-place:
    - `hx-post="/regenerate"`, `hx-target="#sum-<id>"`, `hx-swap="outerHTML"`.
  - Button has a loading state (text changes to “Regenerating…” and disables after 10ms).
- htmx client script is added to page headers.

## Observed Problem
- Clicking “Regenerate summary” shows the loading state, but no network request is recorded (browser devtools) and no server logs appear. The in‑place update does not happen.
- This behavior persists even after adding the htmx client and a plain-HTML fallback (`formaction`/`formmethod`).

## Likely Cause (to verify later)
- Nested forms: the results page content (for download/selection) is wrapped in a large `<form>`. The “Regenerate summary” was added as an inner `<form>` per paper.
  - Nested forms are invalid HTML and browsers don’t reliably fire a `submit` event on the inner form.
  - htmx with `hx-post` on a `<form>` listens to the form’s submit event; if the inner form’s submit doesn’t fire, there’s no htmx request.
  - The button’s `formaction`/`formmethod` fallback can also be ignored by the browser when the inner form is invalid.
- Important: fixing the invalid HTML structure (removing the nested form) should be our first step next time. Once the regenerate control is not nested inside the outer form, either htmx or a plain POST will work reliably. The htmx script itself isn’t the root issue; the markup structure is.
- The immediate-disable-after-10ms pattern is unlikely to be the cause; the same pattern is used on other working buttons.

## Suggested Fix Options (next pass)
Pick one of the following to avoid relying on a nested `<form>` submit.

1) Move htmx to the button and remove the inner form
- Put `hx-post` on the button itself and include the needed fields:
  - `hx-post="/regenerate"`
  - `hx-vals='{"arxiv_id":"<id>","summary_style": "<style>"}'` (or `hx-include` targeting hidden inputs)
  - `hx-target="#sum-<id>"` and `hx-swap="outerHTML"`
- Keep the loading state as-is. This avoids form submit entirely.

2) Keep a form, but take it out of the outer form
- Place the regenerate form outside the large results `<form>` (e.g., wrap each paper card in its own container not contained by the parent form). This guarantees a valid `<form>` submit and htmx intercept.
- Alternatively, associate the button with its form via the HTML `form` attribute if DOM rearrangement is needed.

3) Use a div trigger with htmx
- Wrap the controls in a `<div hx-post="/regenerate" hx-target="#sum-<id>" hx-swap="outerHTML">` and add hidden inputs (`<input name=...>`). Clicking the contained button triggers the htmx request without relying on a form submit.

4) Last-resort fallback
- If htmx remains problematic, use a small inline `fetch` and `outerHTML` replacement. Keep this as a fallback; htmx should be sufficient once the nested-form issue is removed.

## Debugging Tips
- Inspect the DOM to confirm whether the regenerate control is a descendant of the outer results `<form>`.
- Temporarily add `hx-ext='debug'` or set `hx-on` handlers to log events.
- Test a minimal prototype: a single `<button hx-post="/regenerate" hx-vals='{"arxiv_id":"..."}' hx-target="#sum-..." hx-swap="outerHTML">`. If this works outside the outer form, the nested form is the culprit.
- Watch the Network tab while clicking. If no request appears, the trigger didn’t fire; if there’s a request but no swap, verify the returned fragment matches the expected target wrapper id.

## Summary
- Desired UX: live, in-place summary replacement.
- Current state: route and fragment rendering exist; htmx client is loaded; button shows loading; no request made.
- Most probable blocker: nested forms preventing the inner form’s submit/htmx interception.
- Next step: move to a button-based htmx trigger (no nested form) or refactor the regenerate control outside the parent form.
