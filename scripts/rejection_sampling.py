#!/usr/bin/env python
"""Rejection sampling: generate reasoning traces with the BASE model, keep
only traces whose \\boxed{} answer matches gold. Incremental + resumable.

vLLM, temperature 0.9 (configurable 0.8–1.0), n=8 samples per problem,
max 4096 new tokens (longer traces are useless for 4096-seq-len SFT anyway).

Usage (on A100, fresh runtime — no Unsloth loaded):
  python scripts/rejection_sampling.py \
      --data data/train_split.csv data/synth.csv \
      --out /content/drive/MyDrive/nemotron_comp/traces.jsonl

Output jsonl per problem: {id, type, prompt, gold, traces: [..correct only..],
n_correct, n_sampled}. Problems with n_correct == 0 are still written (empty
traces list) — they are the difficulty signal; collect them with:
  jq -c 'select(.n_correct==0)' traces.jsonl
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring import extract_answer, is_correct  # noqa: E402

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    rows = []
    for path in args.data:
        rows.extend(csv.DictReader(open(path, newline="", encoding="utf-8")))

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(ln)["id"] for ln in f if ln.strip()}
        print(f"resuming: {len(done)} problems already sampled")
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print("nothing to do")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(model=MODEL, trust_remote_code=True, max_model_len=8192,
              max_num_seqs=64, gpu_memory_utilization=0.85)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                        max_tokens=args.max_tokens)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for i in range(0, len(todo), args.batch):
        chunk = todo[i: i + args.batch]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        outs = llm.generate(prompts, sp)
        with open(args.out, "a", encoding="utf-8") as f:
            for row, out in zip(chunk, outs):
                correct = []
                for o in out.outputs:
                    if len(o.token_ids) >= args.max_tokens:
                        continue  # truncated, no \boxed{} guaranteed
                    if is_correct(extract_answer(o.text), row["answer"]):
                        correct.append(o.text)
                correct.sort(key=len)  # shortest-first; builder takes [0]
                f.write(json.dumps({
                    "id": row["id"], "type": row.get("type", "?"),
                    "prompt": row["prompt"], "gold": row["answer"],
                    "n_sampled": len(out.outputs), "n_correct": len(correct),
                    "traces": correct[:2],  # keep top-2 shortest to save disk
                }, ensure_ascii=False) + "\n")
        print(f"sampled {min(i + args.batch, len(todo))}/{len(todo)}")

    # quick summary
    from collections import Counter, defaultdict
    stats = defaultdict(Counter)
    with open(args.out, encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            stats[rec["type"]]["total"] += 1
            stats[rec["type"]]["has_trace" if rec["n_correct"] else "zero"] += 1
    print(f"\n{'type':<14}{'total':>7}{'has_trace':>11}{'zero':>7}")
    for t in sorted(stats):
        s = stats[t]
        print(f"{t:<14}{s['total']:>7}{s['has_trace']:>11}{s['zero']:>7}")


if __name__ == "__main__":
    main()
