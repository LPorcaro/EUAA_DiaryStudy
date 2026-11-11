"""Checks for src/rounding.py. Run with ``python tests/test_rounding.py``."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rounding import format_half_up, round_half_up  # noqa: E402


def test_round_half_up() -> None:
    assert round_half_up(3.25, 1) == 3.3          # built-in round gives 3.2
    assert round_half_up(0.235, 2) == 0.24        # stored as 0.23499999...
    assert round_half_up(2.845, 2) == 2.85
    assert round_half_up(37 / 13, 1) == 2.8       # 2.846..., no double rounding
    assert round_half_up(-0.125, 2) == -0.13      # halves away from zero
    assert round_half_up(0.4031, 2) == 0.40
    assert math.isnan(round_half_up(float("nan"), 2))


def test_format_half_up() -> None:
    assert format_half_up(0.235, 2) == "0.24"
    assert format_half_up(0.1, 2) == "0.10"
    assert format_half_up(1, 2) == "1.00"


if __name__ == "__main__":
    test_round_half_up()
    test_format_half_up()
    print("rounding tests passed")
