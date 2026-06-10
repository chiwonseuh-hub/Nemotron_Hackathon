#!/usr/bin/env python
"""Build the SFT dataset from rejection-sampling traces.

- one record per problem: shortest correct trace
- optional general reasoning data mix-in (10–15%) to preserve base ability
- output: jsonl with {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

The assistant turn is the raw sampled trace (already in the model's native
reasoning -> \\boxed{} format, since it was produced by the base model with
the reasoning chat template). We deliberately do NOT rewrite traces — keeping
the native format avoids distribution shift.

Usage:
  python scripts/build_sft_dataset.py \
      --traces /content/drive/MyDrive/nemotron_comp/traces.jsonl \
      --out /content/drive/MyDrive/nemotron_comp/sft_train.jsonl \
      [--general-mix /path/to/general.jsonl --general-frac 0.12]

For --general-mix, prepare a jsonl of {"messages": [...]} records, e.g. a
subsample of nvidia/OpenMathInstruct-2 or similar, formatted the same way.
"""
import argparse
import json
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--general-mix", default=None)
    ap.add_argument("--general-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ids", default=None,
                    help="csv whose ids must be EXCLUDED (val leakage guard)")
    args = ap.parse_args()

    exclude = set()
    if args.val_ids:
        import csv
        exclude = {r["id"] for r in csv.DictReader(open(args.val_ids, encoding="utf-8"))}

    best = {}
    counts = Counter()
    for path in args.traces:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                if not rec["traces"] or rec["id"] in exclude:
                    continue
                trace = rec["traces"][0]  # shortest correct
                if rec["id"] not in best or len(trace) < len(best[rec["id"]][1]):
                    best[rec["id"]] = (rec, trace)

    records = []
    for rec, trace in best.values():
        counts[rec["type"]] += 1
        records.append({"messages": [
            {"role": "user", "content": rec["prompt"]},
            {"role": "assistant", "content": trace}]})

    if args.general_mix:
        n_gen = int(len(records) * args.general_frac / (1 - args.general_frac))
        with open(args.general_mix, encoding="utf-8") as f:
            pool = [json.loads(ln) for ln in f if ln.strip()]
        rng = random.Random(args.seed)
        picked = rng.sample(pool, min(n_gen, len(pool)))
        records.extend(picked)
        counts["general"] = len(picked)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records -> {args.out}")
    for t, n in counts.most_common():
        print(f"  {t:<14}{n}")


if __name__ == "__main__":
    main()
