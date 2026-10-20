#!/usr/bin/env python3
"""Standalone correctness runner (small-network checks are CPU-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sire import correctness  # noqa: E402

res = correctness.small_network_checks()
failed = [r for r in res if not r["passed"]]
for r in res:
    print(("PASS " if r["passed"] else "FAIL ") + r["name"])
print(f"{len(res) - len(failed)}/{len(res)} passed")
sys.exit(0 if not failed else 1)
