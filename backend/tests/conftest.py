"""Shared pytest fixtures for the Looma test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``app`` package importable as a top-level module so tests can do
# ``from app...`` without relying on a particular cwd. This matches the
# layout in backend/pyproject.toml and the run.sh launcher.
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
