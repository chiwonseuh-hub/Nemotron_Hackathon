#!/usr/bin/env python
"""Package the trained LoRA adapter into submission.zip and verify it.

Checks:
  - adapter_config.json present, r <= 32
  - adapter weights file present (adapter_model.safetensors)
  - no oversized junk (optimizer states, checkpoints) inside the zip
After zipping, ALWAYS verify with a real vLLM load in a fresh runtime:
  python scripts/eval_vllm.py --lora <adapter_dir> --data data/val_split.csv --batch 16

Usage:
  python scripts/package_submission.py --adapter <dir> --out submission.zip
"""
import argparse
import json
import os
import zipfile

REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]
EXCLUDE_PREFIXES = ("optimizer", "scheduler", "rng_state", "trainer_state",
                    "training_args", "checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()

    cfg_path = os.path.join(args.adapter, "adapter_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    r = cfg.get("r")
    assert r is not None and r <= 32, f"LoRA rank {r} exceeds 32!"
    print(f"adapter_config OK: r={r}, alpha={cfg.get('lora_alpha')}, "
          f"targets={cfg.get('target_modules')}")

    files = sorted(os.listdir(args.adapter))
    for req in REQUIRED:
        assert req in files, f"missing {req} in {args.adapter}"

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            path = os.path.join(args.adapter, name)
            if not os.path.isfile(path):
                continue
            if name.startswith(EXCLUDE_PREFIXES):
                print(f"  skipping {name}")
                continue
            z.write(path, arcname=name)
            print(f"  + {name} ({os.path.getsize(path) / 1e6:.1f} MB)")

    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print("FINAL GATE: load this adapter with eval_vllm.py --lora in a fresh "
          "runtime and run a few sample inferences before submitting.")


if __name__ == "__main__":
    main()
