"""LoRA fine-tune a small open code model as a binary "is this comment worth posting?" classifier.

Requires: pip install torch transformers peft accelerate   (a free Colab/Kaggle GPU is enough)

    python -m finetune.train_lora --train finetune/data/train.jsonl --val finetune/data/val.jsonl \
        --base Qwen/Qwen2.5-Coder-0.5B --out models/triage-lora

Then set TRIAGE_MODEL_DIR=models/triage-lora to use it as a filter in the review pipeline.
Always compare against the prompted-LLM baseline (see README, "Fine-tuning") before claiming a win.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prf(preds: list[int], labels: list[int]) -> dict:
    tp = sum(1 for p, y in zip(preds, labels, strict=True) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels, strict=True) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels, strict=True) if p == 0 and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = sum(1 for p, y in zip(preds, labels, strict=True) if p == y) / max(len(labels), 1)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--out", default="models/triage-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    train_rows, val_rows = load_jsonl(args.train), load_jsonl(args.val)
    print(f"train={len(train_rows)} val={len(val_rows)} "
          f"positive_rate={sum(r['label'] for r in train_rows) / max(len(train_rows), 1):.2f}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ))
    model.print_trainable_parameters()

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            enc = tok(self.rows[i]["text"], truncation=True, max_length=args.max_len)
            enc["labels"] = int(self.rows[i]["label"])
            return enc

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return prf(np.argmax(logits, axis=-1).tolist(), labels.tolist())

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            eval_strategy="epoch", save_strategy="no", logging_steps=10, report_to="none",
        ),
        train_dataset=DS(train_rows), eval_dataset=DS(val_rows),
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
