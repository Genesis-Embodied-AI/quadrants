#!/usr/bin/env python

import sys
from pathlib import Path

path = Path(__file__).resolve().parent / ".github" / "workflows" / "scripts"
sys.path.insert(0, str(path))

import qd_build.entry

sys.exit(qd_build.entry.main())
