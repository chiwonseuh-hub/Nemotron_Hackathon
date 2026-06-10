#!/usr/bin/env python
"""Stratified 90/10 split of train.csv by detected puzzle type (seed fixed).

Usage: python scripts/split_train_val.py [--data data/train.csv] [--out-dir data]
Writes data/train_split.csv and data/val_split.csv with an extra 'type' column.
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solvers.router import detect_type  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    by_type = defaultdict(list)
    for row in csv.DictReader(open(args.data, newline="", encoding="utf-8")):
        row["type"] = detect_type(row["prompt"])
        by_type[row["type"]].append(row)

    rng = random.Random(SEED)
    train, val = [], []
    for t, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        k = max(1, int(len(rows) * args.val_frac))
        val.extend(rows[:k])
        train.extend(rows[k:])
        print(f"{t:<12} total={len(rows):>6} val={k}")

    fields = ["id", "prompt", "answer", "type"]
    for name, rows in (("train_split.csv", train), ("val_split.csv", val)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
