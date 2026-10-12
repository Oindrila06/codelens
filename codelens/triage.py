"""Optional fine-tuned triage model: predicts whether a comment is worth posting."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def triage_text(c: dict) -> str:
    """Shared by training-data preparation and inference so the formats always match."""
    return (
        f"category: {c.get('category', '')}\nseverity: {c.get('severity', '')}\n"
        f"source: {c.get('source', '')}\ntitle: {c.get('title', '')}\n"
        f"explanation: {str(c.get('explanation', ''))[:800]}\n"
        f"evidence: {str(c.get('evidence', ''))[:600]}"
    )


class TriageModel:
    """Loads a LoRA adapter trained by finetune/train_lora.py. Needs torch + transformers + peft."""

    def __init__(self, model_dir: str) -> None:
        import torch
        from peft import AutoPeftModelForSequenceClassification
        from transformers import AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.config.pad_token_id = self.tok.pad_token_id

    def predict_proba(self, comment: dict) -> float:
        enc = self.tok(triage_text(comment), return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]
        return float(self.torch.softmax(logits, dim=-1)[1])


def load_triage(model_dir: str):
    if not model_dir:
        return None
    try:
        return TriageModel(model_dir)
    except Exception as exc:  # noqa: BLE001 - triage is optional; never break reviews over it
        log.warning("Could not load triage model from %s: %s", model_dir, exc)
        return None
