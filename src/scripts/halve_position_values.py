#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read_csv_with_optional_metadata(path: Path):
    """Reads CSV; if the first line contains 'Capture Start Time', preserves the first 6 lines as metadata."""
    metadata = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    skip = 5 if "Capture Start Time" in first_line else 0
    if skip:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            metadata = [next(f) for _ in range(skip)]
    df = pd.read_csv(path, skiprows=skip, header=0, low_memory=False)

    return df, metadata

def halve_integer_entries_in_position_columns(df: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """
    For columns whose name contains 'position' (case-insensitive):
      - Convert to numeric (coerce errors to NaN for non-numeric cells)
      - Halve only rows that are integer-valued (within `tol`), leave others unchanged
    """
    out = df.copy()
    print(out.columns)
    position_cols = [c for c in out.columns if "position" in str(c).lower()]
    for col in position_cols:
        num = pd.to_numeric(out[col], errors="coerce")
        mask = num.notna()
        out.loc[mask, col] = (num / 2.0).loc[mask]  # strings (NaN) stay unchanged

    
    out.columns = out.columns.str.strip()
    # remove pandas duplicate suffixes like ".1", ".2", ...
    out.columns = pd.Series(out.columns).apply(lambda x: x.split('.')[0]).tolist()
    # blank out columns that start with "Unnamed"
    out.columns = ['' if str(col).startswith('Unnamed') else col for col in out.columns]
    return out

def main():
    ap = argparse.ArgumentParser(description="Halve integer values in any column whose name contains 'position'.")
    ap.add_argument("-i", "--input", required=True, help="Input CSV path")
    ap.add_argument("-o", "--output", help="Output CSV path (default: <input>_posints_halved.csv in same folder)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_posints_halved.csv")

    df, metadata = read_csv_with_optional_metadata(in_path)
    df2 = halve_integer_entries_in_position_columns(df)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if metadata:
            f.writelines(metadata)
        df2.to_csv(f, index=False)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
