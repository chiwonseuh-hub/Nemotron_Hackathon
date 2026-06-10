#!/usr/bin/env python
"""Evaluation harness reproducing the Kaggle spec exactly.

vLLM params: max_lora_rank=32, max_tokens=7680, temperature=0.0, top_p=1.0,
max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.85.
Scoring: \\boxed{} first, last-number fallback; exact or numeric-tolerance match.

Run on A100 80GB. Do NOT run in the same session as Unsloth training —
restart the runtime between training and evaluation.

Usage:
  python scripts/eval_vllm.py --data data/val_split.csv \
      [--lora /path/to/adapter] [--out artifacts/eval_base.jsonl]

Incremental: generations are appended to --out per batch; rerunning skips
already-evaluated ids, so an interrupted Colab session can resume.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring import extract_answer, is_correct  # noqa: E402

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def build_prompts(rows, tokenizer):
    prompts = []
    for row in rows:
        msgs = [{"role": "user", "content": row["prompt"]}]
        # reasoning flag ON (Nemotron template defaults to thinking enabled;
        # keep explicit in case the default changes)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/val_split.csv")
    ap.add_argument("--lora", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--out", default="artifacts/eval.jsonl")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.data, newline="", encoding="utf-8")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(ln)["id"] for ln in f if ln.strip()}
        print(f"resuming: {len(done)} already evaluated")
    todo = [r for r in rows if r["id"] not in done]

    if todo:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        llm = LLM(model=MODEL, trust_remote_code=True,
                  max_model_len=8192, max_num_seqs=64,
                  gpu_memory_utilization=0.85,
                  enable_lora=args.lora is not None, max_lora_rank=32)
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
        lora_req = LoRARequest("adapter", 1, args.lora) if args.lora else None

        for i in range(0, len(todo), args.batch):
            chunk = todo[i: i + args.batch]
            prompts = build_prompts(chunk, tokenizer)
            outs = llm.generate(prompts, sp, lora_request=lora_req)
            with open(args.out, "a", encoding="utf-8") as f:
                for row, out in zip(chunk, outs):
                    gen = out.outputs[0].text
                    f.write(json.dumps({
                        "id": row["id"], "type": row.get("type", "?"),
                        "gold": row["answer"], "pred": extract_answer(gen),
                        "gen_tokens": len(out.outputs[0].token_ids),
                        "generation": gen}, ensure_ascii=False) + "\n")
            print(f"evaluated {min(i + args.batch, len(todo))}/{len(todo)}")

    # ---- report ----
    type_of = {r["id"]: r.get("type", "?") for r in rows}
    gold_of = {r["id"]: r["answer"] for r in rows}
    stats = defaultdict(Counter)
    with open(args.out, encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            if rec["id"] not in gold_of:
                continue
            t = type_of[rec["id"]]
            stats[t]["total"] += 1
            if is_correct(rec["pred"], gold_of[rec["id"]]):
                stats[t]["correct"] += 1
    total = sum(s["total"] for s in stats.values())
    correct = sum(s["correct"] for s in stats.values())
    print(f"\n{'type':<14}{'total':>7}{'correct':>9}{'acc':>8}")
    for t in sorted(stats):
        s = stats[t]
        print(f"{t:<14}{s['total']:>7}{s['correct']:>9}{s['correct'] / s['total']:>8.3f}")
    print(f"{'ALL':<14}{total:>7}{correct:>9}{correct / max(total, 1):>8.3f}")


if __name__ == "__main__":
    main()
