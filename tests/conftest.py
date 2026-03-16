"""pytest configuration.

Adds the repository root to sys.path so that the src package is importable
when running pytest without pip install -e . first. If you do install with
pip install -e .[test], this file is harmless but no longer necessary.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
