#!/usr/bin/env python
"""Create a dummy r=32 LoRA adapter (zero-init B => identity behavior) to
verify the end-to-end vLLM load + submission packaging BEFORE training.

Run on the A100 Colab (loads the 30B model in 4-bit, ~20GB). After this,
run:  python scripts/eval_vllm.py --lora <out> --data data/val_split.csv --batch 16
(in a FRESH runtime) to confirm vLLM accepts the adapter.

Usage: python scripts/make_dummy_lora.py --out /content/drive/MyDrive/nemotron_comp/dummy_lora
"""
import argparse

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

# Standard attention/MLP projections; MoE router modules are intentionally
# excluded (training the router is forbidden by our own constraint and is
# also Unsloth's default exclusion).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16)

    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                     target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    peft_model.save_pretrained(args.out)
    print(f"dummy adapter saved -> {args.out}")
    print("now restart the runtime and run eval_vllm.py --lora to verify loading")


if __name__ == "__main__":
    main()
