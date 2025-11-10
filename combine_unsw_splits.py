#!/usr/bin/env python3
"""
Combine the four UNSW-NB15 split CSV files into a single dataset.

The script expects the raw CSV parts (UNSW-NB15_1.csv … UNSW-NB15_4.csv) to live
inside a folder called ``Datasets`` by default. It attaches the official column
names, concatenates every file in lexical order, drops high-cardinality
identifier columns, and writes the merged output back to disk.

Usage:
    python3 combine_unsw_splits.py
    python3 combine_unsw_splits.py --input-dir ./Datasets --output ./Datasets/unsw_combined.csv
    python3 combine_unsw_splits.py --drop-columns srcip dstip attack_cat
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Column names provided by the official UNSW-NB15 feature list.
COLUMN_NAMES: List[str] = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "label",
]

# Default identifier-style columns to remove after the merge.
DEFAULT_DROP_COLS = ["srcip", "dstip", "Stime", "Ltime", "attack_cat"]


def _default_dataset_dir() -> Path:
    """Pick a sensible default for the raw CSV folder."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "Datasets",
        script_dir.parent / "Datasets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


DEFAULT_INPUT_DIR = _default_dataset_dir()
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "UNSW_NB15_combined.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine UNSW-NB15 split CSV files into a single dataset."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory that holds UNSW-NB15_*.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern",
        default="UNSW-NB15_*.csv",
        help="Glob pattern (relative to input-dir) for the split files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the merged CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        metavar="COL",
        help="Columns to drop after merging. Defaults to identifier columns if omitted.",
    )
    return parser.parse_args()


def discover_parts(input_dir: str, pattern: str) -> List[str]:
    search_pattern = os.path.join(os.path.abspath(input_dir), pattern)
    parts = sorted(glob.glob(search_pattern))
    if not parts:
        raise FileNotFoundError(f"No CSV files matched {search_pattern!r}")
    return parts


def load_split(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        encoding="utf-8-sig",
        low_memory=False,
    )


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols_to_drop = [col for col in columns if col in df.columns]
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("No matching columns to drop.")
    return df


def main() -> None:
    args = parse_args()

    parts = discover_parts(args.input_dir, args.pattern)
    print(f"Found {len(parts)} split file(s):")
    for part in parts:
        print(f"   - {part}")

    frames = []
    for path in parts:
        print(f"Loading {path} ...")
        frames.append(load_split(path))

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined shape: {combined.shape}")

    drop_cols = args.drop_columns if args.drop_columns is not None else DEFAULT_DROP_COLS
    combined = drop_columns(combined, drop_cols)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved merged dataset to: {out_path}")


if __name__ == "__main__":
    main()
