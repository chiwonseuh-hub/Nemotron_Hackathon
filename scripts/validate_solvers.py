#!/usr/bin/env python
"""Step 1 gate: run all solvers over data/train.csv and report per-type
accuracy. Target: >= 99% overall before moving on.

Usage:
  python scripts/validate_solvers.py [--data data/train.csv] [--limit N]

Outputs:
  - per-type accuracy + routing coverage table (stdout)
  - failures dumped to artifacts/solver_failures.jsonl for inspection
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring import is_correct  # noqa: E402
from solvers.router import detect_type, solve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/solver_failures.jsonl")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.data, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stats = defaultdict(Counter)
    failures = []
    for row in rows:
        prompt, gold = row["prompt"], row["answer"]
        ptype = detect_type(prompt)
        try:
            _, pred = solve(prompt, ptype if ptype != "unknown" else None)
        except Exception as e:  # solver bug: count as failure, keep going
            pred = None
            failures.append({"id": row.get("id"), "type": ptype,
                             "error": repr(e), "prompt": prompt, "gold": gold})
        ok = pred is not None and is_correct(pred, gold)
        stats[ptype]["total"] += 1
        stats[ptype]["solved" if pred is not None else "no_answer"] += 1
        stats[ptype]["correct" if ok else "wrong"] += 1
        if not ok:
            failures.append({"id": row.get("id"), "type": ptype,
                             "pred": pred, "gold": gold, "prompt": prompt})

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(s["total"] for s in stats.values())
    correct = sum(s["correct"] for s in stats.values())
    print(f"\n{'type':<12}{'total':>8}{'answered':>10}{'correct':>10}{'acc':>8}")
    for t in sorted(stats):
        s = stats[t]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{t:<12}{s['total']:>8}{s['solved']:>10}{s['correct']:>10}{acc:>8.3f}")
    print(f"{'ALL':<12}{total:>8}{'':>10}{correct:>10}{correct / max(total, 1):>8.3f}")
    print(f"\n{len(failures)} failures -> {args.out}")
    if total and correct / total < 0.99:
        print("BELOW 99% GATE — inspect failures before proceeding.")


if __name__ == "__main__":
    main()
