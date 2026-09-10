"""
src/bert_encoder.py

Wraps a HuggingFace BERT/DistilBERT model to map tokenized lyrics / tags /
MusicCaps captions to contextual embeddings, per the spec:

    H_text = BERT(X_text) in R^{L x d},   t = H_text[CLS]

Used directly by Task 1 (BERT baseline), and as the text tower inside Task 3
(fusion) and Task 4 (contrastive dual encoder).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class BertTextEncoder(nn.Module):
    """Thin wrapper: tokenize -> BERT -> CLS embedding (+ full token states)."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        freeze: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.output_dim = self.encoder.config.hidden_size

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def tokenize(self, texts: List[str], device: Optional[torch.device] = None):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Returns:
            cls_embedding: [B, d]   -- pooled / CLS representation `t`
            token_states:  [B, L, d] -- full sequence H_text, for cross-attention fusion
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state  # [B, L, d]

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_embedding = outputs.pooler_output
        else:
            # DistilBERT has no pooler; use the [CLS] token's hidden state (position 0)
            cls_embedding = token_states[:, 0, :]

        return cls_embedding, token_states

    def encode_texts(self, texts: List[str], device: Optional[torch.device] = None):
        """Convenience one-shot: tokenize + forward. Use for eval / inference,
        not inside a training loop (re-tokenizing every step is wasteful — use
        a cached DataLoader collate_fn there instead, see datasets.py)."""
        enc = self.tokenize(texts, device=device)
        return self.forward(enc["input_ids"], enc["attention_mask"])


class BertTagClassifier(nn.Module):
    """
    Task 1 (Easy): BERT Baseline for Music Tag Understanding.

        t = BERT_CLS(X_text),   y_hat_k = sigmoid(w_k^T t + b_k)
        L_BERT = -1/K * sum_k [ y_k log y_hat_k + (1-y_k) log(1-y_hat_k) ]
    """

    def __init__(
        self,
        num_tags: int,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        freeze_bert: bool = False,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = BertTextEncoder(model_name=model_name, max_length=max_length, freeze=freeze_bert)
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.text_encoder.output_dim, num_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_embedding, _ = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(self.dropout(cls_embedding))
        return logits  # BCEWithLogitsLoss expects raw logits, not sigmoid(logits)

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_ids, attention_mask))
