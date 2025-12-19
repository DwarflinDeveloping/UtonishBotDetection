import os
import csv
import json
import tempfile
from typing import List, Tuple, Dict, Optional
from chatwatcher import watch_loop

CSV_FILE = "diffs.csv"


def read_csv_as_columns(filename: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read CSV where header is usernames and columns contain contiguous diffs (no blanks between entries
    in a single column). Returns header list and columns list-of-lists. Empty file -> ([], []).
    """
    if not os.path.exists(filename):
        return [], []

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []

        # initialize columns
        columns: List[List[str]] = [[] for _ in header]

        for row in reader:
            # For each cell in the row, if it's non-empty, append to that column.
            for i, col_name in enumerate(header):
                cell = row[i] if i < len(row) else ""
                if cell != "":
                    columns[i].append(cell)

    return header, columns


def write_columns_to_csv(header: List[str], columns: List[List[str]]) -> None:
    """
    Write header and columns to CSV. Columns are top-aligned; rows = max column length.
    Pads missing cells at bottom with empty strings.
    Atomic replace is used.
    """
    max_len = max((len(c) for c in columns), default=0)
    dirn = os.path.dirname(os.path.abspath(CSV_FILE)) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dirn, newline="", encoding="utf-8") as tf:
        writer = csv.writer(tf)
        writer.writerow(header)
        for r in range(max_len):
            row = []
            for col in columns:
                row.append(col[r] if r < len(col) else "")
            writer.writerow(row)
        tempname = tf.name
    os.replace(tempname, CSV_FILE)


def append_diff_for_user(header: List[str], columns: List[List[str]], username: str, diff: float) -> Tuple[List[str], List[List[str]]]:
    """
    Append a diff (float) to the user's column. If the user column doesn't exist, create it.
    Returns possibly-updated (header, columns).
    """
    formatted = f"{diff:.3f}"
    if username in header:
        idx = header.index(username)
        columns[idx].append(formatted)
    else:
        header.append(username)
        columns.append([formatted])
    # Rewrite full CSV so columns remain contiguous (no blanks between values in a column).
    write_columns_to_csv(header, columns)
    return header, columns


def main():
    header, columns = read_csv_as_columns(CSV_FILE)
    state = {}

    for item in watch_loop("-fesy2kdDxo"):
        username = item.author.name
        timestamp = item.timestamp / 1000.0  # ms -> seconds

        last_ts = state.get(username)
        if last_ts is not None:
            diff = timestamp - last_ts
            # append diff to that user's column (create column if necessary)
            header, columns = append_diff_for_user(header, columns, username, diff)
            print(username, diff)
        else:
            # First-seen: do NOT write anything to CSV (no timestamp column, only diffs).
            # Just start tracking the timestamp so next message yields a diff.
            pass

        # update and persist last-timestamp
        state[username] = timestamp


if __name__ == "__main__":
    main()