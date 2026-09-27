#!/usr/bin/env bash
# Start the Voice Forge dashboard at http://127.0.0.1:8765
cd "$(dirname "$0")"
exec .venv/bin/python server.py "$@"
