"""Checks for PB parsing in analyze_events.py.

Run with ``python tests/test_pb_parsing.py`` (headless backend is set here).
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from analyze_events import (  # noqa: E402
    extract_pb_statements,
    map_pb_list,
    parse_pb_selections,
)
from config import PB_DICT, RAW_EVENTS_PATH  # noqa: E402

SB1 = next(k for k, v in PB_DICT.items() if v == ("SB", 1))
CI3 = next(k for k, v in PB_DICT.items() if v == ("CI", 3))


def test_free_text_with_commas_is_not_fragmented() -> None:
    cell = f"{SB1}, Perché ascolto tanti artisti e, dando un occhio, altro, {CI3}"
    found, free = extract_pb_statements(cell)
    assert sorted(found) == sorted([SB1, CI3])
    assert free == "Perché ascolto tanti artisti e, dando un occhio, altro"


def test_empty_cell() -> None:
    assert extract_pb_statements(float("nan")) == ([], "")


def test_same_items_as_comma_split_on_real_data() -> None:
    """Known items must be identical to the previous comma-split parser."""
    df = pd.read_csv(RAW_EVENTS_PATH)
    new_mapped, _ = parse_pb_selections(df)
    for i, row in df.iterrows():
        old = []
        for col in ("PB1", "PB2"):
            if isinstance(row[col], str):
                old += row[col].split(",")
        old_known = [t for t in map_pb_list(old) if t != ("O", 2)]
        assert Counter(old_known) == Counter(new_mapped[i]), i


if __name__ == "__main__":
    test_free_text_with_commas_is_not_fragmented()
    test_empty_cell()
    test_same_items_as_comma_split_on_real_data()
    print("PB parsing tests passed")
