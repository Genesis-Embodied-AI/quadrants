#!/usr/bin/env python3

import sys
from pathlib import Path

path = Path(__file__).resolve().parent / ".github" / "workflows" / "scripts"
sys.path.insert(0, str(path))

import qd_build.llvm

sys.exit(qd_build.llvm.main())
