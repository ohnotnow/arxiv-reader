#!/bin/bash

uv run uvicorn main:app --reload --port 8000 --reload-exclude 'papers/' --reload-exclude '.chroma/' --reload-exclude 'state.json' --reload-exclude 'file_index.json'
