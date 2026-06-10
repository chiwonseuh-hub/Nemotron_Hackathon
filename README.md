# NVIDIA Nemotron Model Reasoning Challenge — SFT Pipeline

Goal: LoRA adapter (rank ≤ 32) for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
packaged as `submission.zip`. Strategy: programmatic solvers → verified
synthetic data → rejection-sampled traces → Unsloth QLoRA SFT.

## Layout

```
scoring.py                  \boxed{} extraction + matching (mirrors Kaggle metric)
solvers/                    rule-induction solvers for the 6 puzzle types
generators/gen_puzzles.py   verified synthetic generators (+ OOD variants)
templates/*.txt             prompt templates — PLACEHOLDERS, replace from train.csv
scripts/
  validate_solvers.py       step 1 gate: ≥99% solver accuracy on train.csv
  extract_templates.py      print real prompts per type to build templates
  split_train_val.py        stratified 90/10 split (seed 42)
  make_synthetic.py         generate synthetic problem CSVs (in-dist + OOD)
  make_dummy_lora.py        r=32 dummy adapter for end-to-end vLLM load test
  eval_vllm.py              eval harness, exact Kaggle vLLM params, resumable
  rejection_sampling.py     base-model trace sampling (n=8, T=0.9), resumable
  build_sft_dataset.py      shortest-correct-trace dataset + general mix-in
  train_sft.py              Unsloth QLoRA r=32 α=64, checkpoints to Drive
  package_submission.py     zip + rank/file checks
```

## Colab run order (A100 80GB)

> vLLM and Unsloth cannot coexist in one runtime. Restart between phases.
> All artifacts go to `/content/drive/MyDrive/nemotron_comp/`.

1. **Data in repo**: put `train.csv`/`test.csv` under `data/`.
2. `python scripts/validate_solvers.py` — fix parsers until ≥99%.
   Use `scripts/extract_templates.py` to overwrite `templates/*.txt` with the
   real prompt templates (generators depend on this!).
3. `python scripts/split_train_val.py`
4. *(vLLM runtime)* `python scripts/eval_vllm.py --data data/val_split.csv
   --out .../eval_base.jsonl` — base model per-type baseline.
5. *(transformers runtime)* `python scripts/make_dummy_lora.py --out .../dummy_lora`
   → *(fresh vLLM runtime)* `eval_vllm.py --lora .../dummy_lora --batch 16` —
   end-to-end adapter loading gate.
6. `python scripts/make_synthetic.py --counts <weak types weighted> --out data/synth.csv`
   and `--ood 1500 --out data/ood_synth.csv` (10–20% of total) plus
   `--ood 200 --out data/ood_val.csv` (held-out OOD val).
7. *(vLLM runtime)* `python scripts/rejection_sampling.py --data
   data/train_split.csv data/synth.csv data/ood_synth.csv --out .../traces.jsonl`
8. `python scripts/build_sft_dataset.py --traces .../traces.jsonl --out
   .../sft_train.jsonl --val-ids data/val_split.csv
   [--general-mix .../general.jsonl --general-frac 0.12]`
9. *(Unsloth runtime)* `python scripts/train_sft.py --data .../sft_train.jsonl
   --out .../runs/r32_lr1e4_ep1 --lr 1e-4 --epochs 1`
10. *(fresh vLLM runtime)* eval adapter on `val_split` AND `ood_val` —
    if OOD val drops below the base model, retrain with lower lr/epochs.
11. Iterate 6–10 (max 2–3 rounds), weighting synthetic counts toward weak types.
12. `python scripts/package_submission.py --adapter .../adapter --out submission.zip`
    then the mandatory fresh-runtime vLLM load + sample inference.

Log every run in `experiments.md`.
