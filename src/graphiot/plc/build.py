"""Compile an IEC Structured Text program into a loadable shared library.

Pipeline: ``strucpp program.st -> generated.cpp/.hpp`` then
``g++ generated.cpp shim.cpp -> lib<name>.so``. The result is cached by a hash
of (ST source + shim + strucpp version + compiler flags), so repeated loads are
free. This is the *same* STruC++ compiler the live OpenPLC Runtime v4 uses, so
the twin runs the exact compiled logic of the deployed PLC.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[2]  # project root (src/graphiot/plc -> project)
_THIRD_PARTY = _ROOT / "third_party" / "strucpp"
_SHIM = _PKG / "shim.cpp"

_CXXFLAGS = ["-std=c++17", "-O1", "-fPIC", "-shared", "-fno-gnu-unique"]


@lru_cache(maxsize=1)
def strucpp_version() -> str:
    """The single pinned STruC++ version directory under third_party/strucpp/."""
    env = os.environ.get("GRAPHIOT_STRUCPP_VERSION")
    if env:
        return env
    versions = sorted(p.name for p in _THIRD_PARTY.iterdir() if (p / "strucpp").exists())
    if not versions:
        raise FileNotFoundError(f"no STruC++ toolchain under {_THIRD_PARTY}")
    return versions[-1]


def strucpp_paths() -> tuple[Path, Path]:
    base = _THIRD_PARTY / strucpp_version()
    return base / "strucpp", base / "runtime" / "include"


def _hash(st_source: str) -> str:
    h = hashlib.sha256()
    h.update(st_source.encode())
    h.update(_SHIM.read_bytes())
    h.update(strucpp_version().encode())
    h.update(" ".join(_CXXFLAGS).encode())
    return h.hexdigest()[:16]


def compile_st(st_path: str | Path, *, cache_dir: str | Path | None = None,
               force: bool = False) -> Path:
    """Compile ``st_path`` to a ``.so`` and return its path (cached)."""
    st_path = Path(st_path)
    source = st_path.read_text()
    strucpp_bin, runtime_inc = strucpp_paths()
    cache_dir = Path(cache_dir) if cache_dir else _ROOT / "build" / "plc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{st_path.stem}_{_hash(source)}"
    so_path = cache_dir / f"lib{tag}.so"
    if so_path.exists() and not force:
        return so_path

    work = cache_dir / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        # 1. ST -> C++ (fixed names so the shim can #include "generated.hpp").
        subprocess.run(
            [str(strucpp_bin), str(st_path), "-o", str(work / "generated.cpp")],
            check=True, capture_output=True, text=True,
        )
        # 2. shim + generated -> shared library.
        shutil.copy(_SHIM, work / "shim.cpp")
        subprocess.run(
            ["g++", *_CXXFLAGS, f"-I{work}", f"-I{runtime_inc}",
             str(work / "generated.cpp"), str(work / "shim.cpp"),
             "-o", str(so_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:  # surface compiler output
        raise RuntimeError(
            f"PLC build failed for {st_path.name}:\n"
            f"CMD: {' '.join(exc.cmd)}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
        ) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return so_path
