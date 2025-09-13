# Repository Guidelines

This project is to help people find and read arXiv papers which are relevant to them.  The main features are:

- Search by category (e.g., `cs.CL`) or interest (e.g., `math.GR`).
- Add a specific sub-interest and use chromadb to find papers which are semantically similar.
- Give a summary style prompt which is used to summarise the paper via OpenAI LLM calls.
- Download paper abstracts
- Allow the user to select ones that are interesting to them and summarize them with OpenAI.
- Cache the results of the above to avoid repeated work.

## Project Structure & Module Organization
- `main.py`: FastHTML web app (routes, arXiv fetch, Chroma narrowing, OpenAI summaries).
- `pyproject.toml`: Python 3.13 + deps (uv-managed).
- `papers/`: Run outputs (PDFs, extracted text, summaries). Ignored by Git.
- `state.json`, `file_index.json`, `.chroma/`: Local app state and vector cache. Ignored by Git.
- `screenshots/`: UI references for PRs and docs.

## Build, Test, and Development Commands
- Setup (uv): `uv venv && source .venv/bin/activate && uv sync`
- Run (hot reload): `uv run uvicorn main:app --reload --port 8000`
- Alt run: `python -m uvicorn main:app --reload`
- Fresh state (dev): `FRESH=1 uv run uvicorn main:app --reload`
- Data location: outputs under `papers/<YYYYmmdd-HHMMSS>/`.

## Coding Style & Naming Conventions
- Python: 4-space indent, type hints preferred; keep functions `lower_snake_case`, classes `UpperCamelCase`, constants `UPPER_SNAKE_CASE`.
- Module layout: keep web/UI helpers near route handlers in `main.py`; factor small pure helpers above routes.
- Docstrings for public helpers; short inline comments only where non-obvious.
- Suggested tools (optional): Black + Ruff. If used, run `ruff format` then `ruff check --fix`.

## Testing Guidelines
- No formal test suite yet. Prefer `pytest` with tests in `tests/` named `test_*.py`.
- When adding tests, target pure helpers first (e.g., `_slugify`, time/window logic). Avoid network in unit tests; mock arXiv/OpenAI/Chroma.
- Manual QA: start app, fetch a small category (e.g., `cs.CL`), select 1–2 papers, verify download, text extraction, and summary rendering.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject, scope if useful (e.g., `feat(ui): add previous matches view`).
- PRs: include purpose, key changes, how to test, and screenshots for UI changes (`screenshots/` or new captures). Link related issues.
- Keep PRs focused; avoid unrelated refactors. Note any migration of local data/state.

## Security & Configuration Tips
- Export `OPENAI_API_KEY` to enable summaries; don’t commit secrets. `.gitignore` already excludes local state and outputs.
- Avoid writing outside repo; use `papers/` and `.chroma/`. Be careful with UUID-file serving logic when introducing new file kinds.

## Architecture Overview
- Flow: Select category/interest → fetch recent arXiv → optional semantic narrowing via Chroma → user selects → download PDFs → extract text → summarize with OpenAI → serve files via UUID endpoints.
