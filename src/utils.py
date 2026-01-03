from __future__ import annotations

from typing import Any


def fmt_value(value: Any) -> str:
    if isinstance(value, (int, float)) and value is not None:
        return f"{value:.2f}"
    return str(value)
