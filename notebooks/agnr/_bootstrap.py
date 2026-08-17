"""_bootstrap.py — import shim for the topic-folder layout.

Scripts in this tree were originally all in one flat directory, so they import
each other by bare module name (``import agnr_lib``). After the split into topic
folders those siblings are no longer on ``sys.path``, because Python only adds the
*running script's own* directory.

Any script that imports a module from a different topic folder does::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import _bootstrap  # noqa: F401

which puts every topic folder on ``sys.path`` and keeps the bare imports working.

Also exposes PROJECT_ROOT for scripts that need repo-level data files.
"""

import sys
from pathlib import Path

AGNR_ROOT = Path(__file__).resolve().parent          # notebooks/agnr
PROJECT_ROOT = AGNR_ROOT.parents[1]                  # transmissions/

_TOPIC_DIRS = (
    "physics",
    "concentration",
    "defect_reconstruction",
    "agent",
    "multi_width",
    "sweeps",
)

for _name in _TOPIC_DIRS:
    _p = AGNR_ROOT / _name
    if _p.is_dir():
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)

if str(AGNR_ROOT) not in sys.path:
    sys.path.insert(0, str(AGNR_ROOT))
