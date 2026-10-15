"""Helpers that several validator modules need and that were copied byte for byte into each (M2643: _text x3, _number x2, _squash x3, _is_number x4).

This file is part of the evaluator (integrity.EVALUATOR_GLOBS), so a change here moves the evaluator hash like a change to a validator.
Helpers with the same name but different behavior in other modules (`_text` with a minimum, `_facts`, `_num`, `_keys`) are NOT here on purpose.
"""

from __future__ import annotations

import re
from typing import Any


def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def squash(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()
