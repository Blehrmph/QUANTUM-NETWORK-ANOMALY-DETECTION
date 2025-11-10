#!/usr/bin/env python3
"""
Export only the PCA features (PC1–PC4) plus the attack category.

Accepts either a single CSV file or a directory of CSVs, optionally
recursively. If a directory is provided, all matching CSVs are
concatenated before column selection.

Output columns (in order): PC1, PC2, PC3, PC4, attack_cat
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REQUIRED_COLUMNS: List[str] = ["PC1", "PC2", "PC3", "PC4", "attack_cat"]
# If attack_cat is missing, we try to map common alternatives:
ATTACK_ALIASES: Sequence[str] = ("attack_cat", "attack_catagory", "attack", "label", "y")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = "/Users/mac/Downloads/Folder/Fall26/QC/Q-ADAM/Datasets"
    default_output = script_dir / "UNSW_NB15_PCA_attack.csv"

    p = argparse.ArgumentParser(
        description="Export PC1–PC4 plus attack_cat from PCA CSV(s). "
                    "Pass a file or a directory as --input."
    )
    p.add_argument(
        "--input",
        default=str(default_input),
        help=f"CSV file OR directory containing CSVs (default: {default_input})",
    )
    p.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern when --input is a directory (default: *.csv)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when --input is a directory.",
    )
    p.add_argument(
        "--output",
        default=str(default_output),
        help=f"Output CSV path (default: {default_output})",
    )
    p.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows with missing attack_cat instead of leaving NaNs.",
    )
    return p.parse_args()


def _find_csvs(path: Path, pattern: str, recursive: bool) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"--input not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"--input is a file but not a .csv: {path}")
        return [path]

    # Directory
    if recursive:
        files = [p for p in path.rglob(pattern) if p.is_file()]
    else:
        files = [p for p in path.glob(pattern) if p.is_file()]

    files = [p for p in files if p.suffix.lower() == ".csv"]
    if not files:
        raise FileNotFoundError(
            f"No CSVs found in directory {path} with pattern '{pattern}' "
            f"(recursive={recursive})."
        )
    return sorted(files)


def _unify_attack_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure an 'attack_cat' column exists.
    If missing, try to map from aliases (e.g., 'label' -> 'attack_cat').
    """
    if "attack_cat" in df.columns:
        return df

    for cand in ATTACK_ALIASES:
        if cand in df.columns:
            if cand != "attack_cat":
                df = df.rename(columns={cand: "attack_cat"})
            return df

    raise ValueError(
        f"None of the label columns found: {ATTACK_ALIASES}. "
        f"Available columns: {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}"
    )


def _load_concat_csvs(files: list[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        # Let pandas auto-detect; add engine="c" hint only if needed.
        df = pd.read_csv(f)
        df = _unify_attack_column(df)
        dfs.append(df)
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    files = _find_csvs(in_path, args.pattern, args.recursive)
    print(f"Found {len(files)} file(s). Example: {files[0]}")
    df = _load_concat_csvs(files)

    missing = [col for col in ["PC1", "PC2", "PC3", "PC4"] if col not in df.columns]
    if missing:
        raise ValueError(
            "Input data is missing PCA columns. "
            f"Missing: {missing}. Available columns (first 25): {list(df.columns)[:25]}"
        )

    # Select and re-order
    subset = df[["PC1", "PC2", "PC3", "PC4", "attack_cat"]].copy()

    if args.drop_missing:
        before = len(subset)
        subset = subset.dropna(subset=["attack_cat"])
        dropped = before - len(subset)
        print(f"Dropped {dropped} rows with missing attack_cat.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)
    print(
        f"Wrote {len(subset)} rows with columns {REQUIRED_COLUMNS} to {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()
