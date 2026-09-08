"""Make ``detr_mor`` importable when running these scripts from a clone.

Importing this module puts the repository root on ``sys.path``, so the scripts
work with a plain ``python scripts/train.py`` and no ``pip install -e .``.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
