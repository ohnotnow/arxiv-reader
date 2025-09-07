# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an arXiv Helper application built with FastHTML that helps researchers:
- Browse and filter arXiv papers by category and interest
- Download PDFs and extract text content
- Generate AI summaries using OpenAI GPT models
- Use vector search (ChromaDB) for semantic paper filtering
- Track reading progress with persistent state management

## Core Architecture

### Main Components

- **Single-file FastHTML application** (`main.py`): All application logic in ~900 lines
- **Data storage**: Local file system with JSON state management
- **Vector database**: ChromaDB for semantic search and embedding caching
- **AI integration**: OpenAI API for paper summarization
- **File organization**: `papers/YYYYMMDD-HHMMSS/` timestamped directories

### Key Data Flow

1. User selects arXiv category and optional interest filter
2. App fetches papers since last run using arXiv API
3. Optional: Vector search narrows results using ChromaDB embeddings
4. User selects papers to download
5. PDFs downloaded, text extracted, summaries generated
6. Results stored in timestamped directory with state tracking

### State Management

- `state.json`: Per-category timestamps and user preferences
- `file_index.json`: UUID mapping for served files
- `.chroma/`: ChromaDB vector database storage

## Development Commands

### Environment Setup
```bash
# Install uv if not available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv python install 3.13
uv venv
source .venv/bin/activate  # Windows: .venv/Scripts/activate

# Install dependencies
uv sync
```

### Running the Application
```bash
# Set OpenAI API key (optional - app works without summaries)
export OPENAI_API_KEY=sk-...

# Run development server with auto-reload
uv run uvicorn main:app --reload --port 8000

# Production server
uv run uvicorn main:app --port 8000
```

### Dependency Management
```bash
# Add new dependency
uv add package-name

# Remove dependency
uv remove package-name

# Update lockfile
uv lock

# Sync environment from lockfile
uv sync
```

## Application Structure

### FastHTML Routes
- `GET /`: Main interface with category selection and paper browsing
- `POST /fetch`: Fetch and filter papers from arXiv
- `POST /download`: Download selected papers with PDF/text extraction
- `GET /previous`: Browse previously downloaded papers
- File serving routes: `/file/{type}/{uuid}` for PDFs, text, summaries

### Core Functions

**Data fetching:**
- `arxiv_fetch()`: Async arXiv API integration with rate limiting
- `narrow_with_chroma()`: Vector similarity search for paper filtering

**Content processing:**
- `_pdf_text()`: PDF text extraction using pypdf
- `openai_summarize()`: AI summarization with configurable styles

**State management:**
- `_load_state()/_save_state()`: Persistent JSON state
- `_register_file()`: File UUID mapping for secure serving

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Optional OpenAI API key for summaries
- `TOKENIZERS_PARALLELISM=false`: Silences HuggingFace tokenizer warnings

### arXiv Categories
Predefined mapping in `CATEGORIES` dict:
- cs.AI, cs.CL, cs.CV, cs.LG, cs.RO, cs.IR, stat.ML

### ChromaDB Integration
- Automatic embedding generation for papers
- Persistent storage in `.chroma/` directory
- Used for semantic filtering when interest terms provided

## File Organization

```
papers/
├── YYYYMMDD-HHMMSS/          # Timestamped download sessions
│   ├── {arxiv_id}.txt        # Extracted paper text
│   ├── {arxiv_id}.pdf        # Downloaded PDFs
│   └── {arxiv_id}.summary.txt # AI-generated summaries
state.json                     # Per-category timestamps and preferences
file_index.json               # UUID to file path mapping
.chroma/                      # ChromaDB vector database
```

## Dependencies

Key packages (see `pyproject.toml`):
- `python-fasthtml`: Web framework
- `arxiv`: arXiv API client  
- `pypdf`: PDF text extraction
- `openai`: AI summarization
- `chromadb`: Vector database
- `uvicorn`: ASGI server
- `python-dateutil`: Date parsing

## Development Notes

- No build step required - uses CDN for Tailwind CSS
- No test suite currently implemented
- Application designed for single-user local usage
- All data stored locally with no external database requirements
- Rate limiting implemented for arXiv API compliance