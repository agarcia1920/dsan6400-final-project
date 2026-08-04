"""Shared path setup for analysis notebooks."""

from pathlib import Path
import sys


def project_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src" / "soulcycle_network").exists():
        return root
    if (root.parent / "src" / "soulcycle_network").exists():
        return root.parent
    raise RuntimeError("Could not locate project root (expected src/soulcycle_network).")


def ensure_src_on_path() -> Path:
    root = project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root
