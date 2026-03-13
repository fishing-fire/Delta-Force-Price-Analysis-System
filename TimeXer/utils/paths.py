from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote


def rel_posix(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=Path(base_dir).resolve())).as_posix()


def rel_link(path: Path, base_dir: Path) -> str:
    rel = rel_posix(path, base_dir)
    if not rel.startswith((".", "/")):
        rel = "./" + rel
    return quote(rel, safe="/:._-")
