#!/usr/bin/env python
"""Generate verified synthetic puzzles (in-distribution + OOD variants).

Usage:
  python scripts/make_synthetic.py --counts bits=2000,cipher=2000 --out data/synth.csv
  python scripts/make_synthetic.py --ood 200 --out data/ood_val.csv

Counts for in-distribution types should be weighted by how WEAK the base
model is per type (see eval results) — pass them explicitly via --counts.
--ood N generates N/6 of each OOD neighbor variant (12/16-bit puzzles,
v=g*t, affine conversions, caesar cipher, base-b numerals).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generators.gen_puzzles import GENERATORS, OOD_GENERATORS, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="", help="type=N,comma separated")
    ap.add_argument("--ood", type=int, default=0, help="total OOD items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = []
    if args.counts:
        for part in args.counts.split(","):
            kind, n = part.split("=")
            assert kind in GENERATORS, f"unknown type {kind}"
            plan.append((kind, int(n)))
    if args.ood:
        per = max(1, args.ood // len(OOD_GENERATORS))
        plan.extend((k, per) for k in OOD_GENERATORS)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "answer", "type"])
        i = 0
        for kind, n in plan:
            for _, prompt, ans in generate(kind, n, seed=args.seed + hash(kind) % 10000):
                w.writerow([f"synth-{kind}-{i}", prompt, ans, kind])
                i += 1
            print(f"{kind}: {n} done")
    print(f"wrote {i} rows -> {args.out}")


if __name__ == "__main__":
    main()
