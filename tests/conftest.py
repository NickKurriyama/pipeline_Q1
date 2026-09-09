"""Make the project root and pipeline3d importable from the tests, no install needed."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "pipeline3d")):
    if p not in sys.path:
        sys.path.insert(0, p)
