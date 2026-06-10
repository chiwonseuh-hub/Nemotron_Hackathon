#!/usr/bin/env python
"""SFT with Unsloth QLoRA on NVIDIA-Nemotron-3-Nano-30B-A3B.

Run in a FRESH Colab runtime (A100 80GB) — do not mix with a vLLM session.
Base recipe per plan: r=32, alpha=64, lr 1e-4 (conservative start), 1 epoch,
max_seq_length 4096, train on assistant responses only, checkpoints to Drive
every --save-steps so an interrupted session resumes via --resume.

MoE router layers are NOT touched (Unsloth default; we only target attention
and expert MLP projections).

Follow the official Unsloth Nemotron-3 notebook (docs.unsloth.ai/models/nemotron-3)
for any version-specific install pins.

Usage:
  python scripts/train_sft.py \
      --data /content/drive/MyDrive/nemotron_comp/sft_train.jsonl \
      --out  /content/drive/MyDrive/nemotron_comp/runs/r32_lr1e4_ep1 \
      [--lr 1e-4] [--epochs 1] [--resume]
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        # attention + expert MLP projections; router excluded (Unsloth default)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_dataset("json", data_files=args.data, split="train")

    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=args.out,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
    )

    # train on assistant responses only (mask the prompt tokens)
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer)
    except Exception as e:
        print(f"WARNING: train_on_responses_only unavailable ({e}); "
              "training on full sequences — check Unsloth Nemotron-3 notebook "
              "for the correct instruction/response markers.")

    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(f"{args.out}/adapter")
    tokenizer.save_pretrained(f"{args.out}/adapter")
    print(f"adapter saved -> {args.out}/adapter")


if __name__ == "__main__":
    main()
