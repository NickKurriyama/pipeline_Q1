"""
tools/patch_gsplat.py  --  make gsplat's JIT build work with MSVC on Windows.

gsplat's JIT loader passes GCC-style flags (-O3 -Wno-attributes) to cl.exe, which fails
with 'D8021 invalid numeric argument'. This script patches the installed
gsplat/cuda/_backend.py to use /O2 (or /Od) on Windows. Idempotent; rerun after any
`pip install -U gsplat`.

Usage:  python tools/patch_gsplat.py
"""
from __future__ import annotations
import os

OLD = '''        extra_cflags = [opt_level, "-Wno-attributes"]'''
NEW = '''        if os.name == "nt":
            # MSVC cl.exe: GCC-style flags (-O3, -Wno-attributes) are invalid (D8021)
            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]
        else:
            extra_cflags = [opt_level, "-Wno-attributes"]'''


def main() -> None:
    import gsplat
    path = os.path.join(os.path.dirname(gsplat.__file__), "cuda", "_backend.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if NEW in src:
        print(f"already patched: {path}")
        return
    if OLD not in src:
        raise SystemExit(f"pattern not found in {path} — gsplat version changed; "
                         "patch manually (replace the GCC-style extra_cflags with /O2).")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
