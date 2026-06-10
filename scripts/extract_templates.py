#!/usr/bin/env python
"""Help derive real prompt templates from train.csv.

Prints, for each detected type, a few full example prompts so the fixed
template text and the variable slots ({examples}, {query}) can be identified
and copied into templates/<type>.txt.

Usage: python scripts/extract_templates.py [--data data/train.csv] [--per-type 3]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solvers.router import detect_type  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--per-type", type=int, default=3)
    args = ap.parse_args()

    by_type = defaultdict(list)
    for row in csv.DictReader(open(args.data, newline="", encoding="utf-8")):
        by_type[detect_type(row["prompt"])].append(row)

    for t, rows in sorted(by_type.items()):
        print(f"\n{'=' * 70}\nTYPE: {t}  (n={len(rows)})\n{'=' * 70}")
        for row in rows[: args.per_type]:
            print(f"\n--- id={row.get('id')} answer={row['answer']!r} ---")
            print(row["prompt"])


if __name__ == "__main__":
    main()
