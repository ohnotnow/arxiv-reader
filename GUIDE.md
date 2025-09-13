# Debug-First Collaboration Guide

This guide captures how we work together: debug first, change last, keep fixes small and evidence‑driven, and communicate clearly. It applies across stacks (FastHTML/htmx, Flask/FastAPI, Laravel/Livewire, Go, etc.).

## Principles
- Debug first, change last — no edits until a cause is proven.
- Evidence over assumption — form a hypothesis, test quickly, then act.
- Small, reversible patches — avoid new deps or architectural shifts.
- Keep scope tight to the failing path.
- Ask questions, collaborate, narrate reasoning. If guessing, say so and stop.

## Working Agreement
- Ask for quick, specific checks before proposing code changes.
- Share acceptance criteria (what success looks like) before coding.
- Propose 1–2 minimal options; get confirmation before adding complexity.
- Remove temporary logs after verification.

## Acceptance Criteria (Template)
- Trigger: “When I click <control> …”
- Effect: “… sends 1 request to <route> with params <…> …”
- UI: “… only updates region <selector> … no full reload.”
- Errors: “No console errors; server logs show route hit and params.”

## Client-Side Debug Checklist (htmx/FastHTML friendly)
1) Reproduce with DevTools open (Preserve log, Disable cache).
2) Verify framework presence and version:
   - `window.htmx` (object?) and `htmx.version` if available.
3) Enable logs and click:
   - `window.htmx.logAll()` → expect trigger/config logs.
4) Validate target selector:
   - Copy `hx-target`, run `document.querySelector('<hx-target>')` — must resolve.
   - If not, fix selector/ID (see “Selector/ID Hygiene”).
5) Force processing if needed:
   - `htmx.process(document.body)` then retry.
6) Minimal trigger sanity in Elements panel:
   - Keep only `hx-post`, `hx-target`, `hx-swap`; remove extras; click again.
7) Manual request to isolate attributes vs route:
   - `htmx.ajax('POST','/route',{ target:'#id', values:{...} })`
8) Watch console for errors (CSP, JS exceptions) and Network tab for requests.

## Server-Side Debug Checklist
- Route method matches (GET vs POST) and is reachable.
- Log route entry and parsed params at the top.
- Verify side effects (e.g., file writes) and return value.
- For fragment swaps: returned wrapper id matches `hx-target` element id.
- Return explicit 4xx/5xx with helpful text if inputs missing.

## Selector/ID Hygiene (HTML/CSS)
- Make DOM ids CSS‑selector safe: only `[a-zA-Z0-9_-]`.
- Never embed raw ids containing dots/spaces directly into element ids.
- Sanity check: `document.querySelector('#the-id')` must succeed unescaped.
- Avoid nested forms; prefer a single form or non‑form triggers.
- Keep triggers simple first: `hx-post`, `hx-target`, `hx-swap`; add extras later.

## Version Discipline (htmx and friends)
- Pin to a single known version; don’t mix 1.x vs 2.x behaviors.
- If using a CDN, pin the exact version; vendor locally only if CSP/CDN issues are proven.

## Make Changes (only after proven cause)
- Implement one small, reversible patch tied to the finding.
- Explain the change and how it addresses the root cause.
- Remove temporary diagnostics; keep code tidy.
- Provide a rollback plan if needed.

## Validate Fix
- Confirm acceptance criteria: request count, params, UI swap, no errors.
- Check regression risk in adjacent flows.
- Capture a quick note/screenshot for future reference.

## What I’ll Ask You For (fast collaboration)
- Console output of `window.htmx.logAll()` interaction.
- The control’s outerHTML and exact `hx-target` value.
- Result of `document.querySelector('<hx-target>')`.
- Any server log lines on route entry.
- Any environment quirks (CSP, proxies, blocked CDNs).

## Minimal Options First (when proposing fixes)
- Option A: smallest client‑side tweak (e.g., CSS‑safe id, simple htmx trigger).
- Option B: smallest server‑side tweak (e.g., accept missing param, stable GET fallback).
- Avoid adding deps, fallbacks, or new endpoints unless necessary.

## Notes for Other Stacks
- React/Vue: validate component props/state, event binding, network call; use React DevTools/Vue devtools.
- Laravel/Livewire: check component events, wire:model bindings, network requests, CSRF tokens.
- Go/Net/http: method routing, handler logs, response codes/content type, template regions.
- CLI/tools: reproduce with verbose flags; isolate inputs/outputs in temp dirs.

## Keep It Human
- If something isn’t clear, ask. If I’m guessing, I’ll say so.
- Pause for your direction before expanding scope.
- We value speed through clarity, not code volume.
