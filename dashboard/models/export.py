"""CSV downloads that always carry the data disclaimer (scope §4)."""

from collections.abc import Sequence

import pandas as pd

DEMO_SUFFIX = "_demo"


def csv_with_notes(frame: pd.DataFrame, notes: Sequence[str]) -> bytes:
    """CSV text with each note as a leading ``#`` comment line (the disclaimer first)."""
    header = "".join(f"# {note}\n" for note in notes)
    return (header + frame.to_csv(index=False)).encode("utf-8")


def demo_file_name(stem: str) -> str:
    """File name for a demo download: ``<stem>_demo.csv``."""
    return f"{stem}{DEMO_SUFFIX}.csv"
