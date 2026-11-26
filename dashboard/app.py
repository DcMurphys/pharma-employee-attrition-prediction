"""MurphysBio attrition app. Run from the repository root (scope §8):

    streamlit run dashboard/app.py

The root and ``src/`` are added to ``sys.path`` so the app runs without ``pip install -e .``
(for example on Streamlit Community Cloud).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dashboard.controllers import shell  # noqa: E402  (needs the paths above)

shell.run()
