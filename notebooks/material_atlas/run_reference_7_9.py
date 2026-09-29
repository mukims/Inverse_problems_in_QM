#!/usr/bin/env python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ref_dir = HERE / "reference_7_9"
if str(ref_dir) not in sys.path:
    sys.path.insert(0, str(ref_dir))

from run_reference_7_9 import main

if __name__ == "__main__":
    main()
