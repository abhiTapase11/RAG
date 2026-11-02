"""Shared script prelude: put ``src/`` on sys.path, make printing safe.

Importing this module is all a script in this directory needs to run from a
clean clone, with no install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Citations contain a section sign; never let a narrow console encoding turn a
# printed answer into a crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - non-reconfigurable stream
        pass
