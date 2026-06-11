#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _sanitize_thread_env() -> None:
    """Avoid libgomp warnings when OMP_NUM_THREADS is set to an invalid value.

    Some platforms export non-integer values like "auto" or list-like strings.
    libgomp warns loudly during import-time OpenMP init. We only coerce invalid
    values, and leave valid integers untouched.
    """

    def _coerce_positive_int(v: str) -> str | None:
        s = str(v).strip()
        if not s:
            return None
        head = s.split(",", 1)[0].strip()
        try:
            n = int(head)
            return str(n) if n > 0 else None
        except Exception:
            return None

    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if k not in os.environ:
            continue
        v = _coerce_positive_int(os.environ.get(k, ""))
        if v is None:
            os.environ[k] = "1"
        else:
            os.environ[k] = v


_sanitize_thread_env()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import main


if __name__ == "__main__":
    main()
