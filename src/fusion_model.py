"""
src/fusion_model.py

Task 3 (Hard): GNN-BERT Fusion for Multi-Context Understanding.

    Cross-attention fusion:
        A = softmax( Q K^T / sqrt(d) ),   Q = g W_Q,   K = H_text W_K
        z = CONCAT(g, A H_text),          y_hat = sigmoid(W z)

    Multi-task loss:
        L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2

Also implements the "early concat" ablation (fusion_type="concat") required
by Task 3's ablation deliverable (BERT-only, GNN-only, early concat,
cross-attention).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from src.bert_encoder import BertTextEncoder
from src.gnn_model import GNNEncoder


class CrossAttentionFusion(nn.Module):
    """
    Single-head (generalizable to multi-head) cross-attention where the graph
    embedding `g` attends over the BERT token sequence `H_text`, exactly as
    specified: A = softmax(QK^T / sqrt(d)), z = CONCAT(g, A H_text).
    """

    def __init__(self, gnn_dim: int, bert_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj_dim = proj_dim
        self.w_q = nn.Linear(gnn_dim, proj_dim)
        self.w_k = nn.Linear(bert_dim, proj_dim)
        self.w_v = nn.Linear(bert_dim, proj_dim)
        self.out_dim = gnn_dim + proj_dim

    def forward(
        self, g: torch.Tensor, h_text: torch.Tensor, text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g:        [B, gnn_dim]      graph-level embedding
        h_text:   [B, L, bert_dim]  full BERT token sequence
        text_mask:[B, L] 1 for real tokens, 0 for padding (optional)

        Returns:
            z:     [B, gnn_dim + proj_dim]  fused representation
            attn:  [B, 1, L]  attention weights (for the case-study deliverable:
                   "3 case studies showing graph paths + caption/lyric alignment")
        """
        q = self.w_q(g).unsqueeze(1)           # [B, 1, proj_dim]
        k = self.w_k(h_text)                    # [B, L, proj_dim]
        v = self.w_v(h_text)                    # [B, L, proj_dim]

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.proj_dim ** 0.5)  # [B, 1, L]
        if text_mask is not None:
            scores = scores.masked_fill(text_mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)    # [B, 1, L]

        attended = torch.bmm(attn, v).squeeze(1)  # [B, proj_dim]
        z = torch.cat([g, attended], dim=-1)      # [B, gnn_dim + proj_dim]
        return z, attn


class ConcatFusion(nn.Module):
    """The 'early concat' ablation: z = CONCAT(g, t) with no attention."""

    def __init__(self, gnn_dim: int, bert_dim: int):
        super().__init__()
        self.out_dim = gnn_dim + bert_dim

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([g, t], dim=-1)


class GNNBertFusionModel(nn.Module):
    """
    End-to-end Task-3 model (Algorithm 3):
        H_text, t <- BERT(X_text)
        g         <- GNN readout(G)
        z         <- CrossAttention(g, H_text)   or   CONCAT(g, t)
        y_hat     <- sigmoid(W z + b)
        (v_hat, a_hat) <- separate regression heads, if predict_emotion=True
    """

    def __init__(
        self,
        num_tags: int,
        gnn_in_channels: int,
        gnn_hidden: int = 64,
        gnn_out: int = 64,
        gnn_layers: int = 3,
        gnn_type: str = "graphsage",
        bert_model_name: str = "distilbert-base-uncased",
        fusion_type: Literal["cross_attention", "concat"] = "cross_attention",
        fusion_hidden_dim: int = 256,
        predict_emotion: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn = GNNEncoder(
            in_channels=gnn_in_channels,
            hidden_channels=gnn_hidden,
            out_channels=gnn_out,
            num_layers=gnn_layers,
            encoder=gnn_type,
            dropout=dropout,
        )
        self.text_encoder = BertTextEncoder(model_name=bert_model_name)
        bert_dim = self.text_encoder.output_dim

        self.fusion_type = fusion_type
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(gnn_out, bert_dim, proj_dim=fusion_hidden_dim)
            fused_dim = self.fusion.out_dim
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(gnn_out, bert_dim)
            fused_dim = self.fusion.out_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.tag_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_tags),
        )

        self.predict_emotion = predict_emotion
        if predict_emotion:
            self.emotion_head = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim // 2, 2),  # (valence, arousal)
            )

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        node_h = self.gnn(graph_x, graph_edge_index)
        g = self.gnn.pool(node_h, graph_batch, mode="mean")           # [B, gnn_out]
        t, h_text = self.text_encoder(input_ids, attention_mask)      # t: [B, bert_dim], h_text: [B, L, bert_dim]

        if self.fusion_type == "cross_attention":
            z, attn = self.fusion(g, h_text, text_mask=attention_mask)
        else:
            z, attn = self.fusion(g, t), None

        tag_logits = self.tag_head(z)
        valence_arousal = self.emotion_head(z) if self.predict_emotion else None
        return {
            "tag_logits": tag_logits,
            "valence_arousal": valence_arousal,
            "attention": attn,
            "graph_embedding": g,
            "text_embedding": t,
            "fused_embedding": z,
        }


def fusion_multitask_loss(
    tag_logits: torch.Tensor,
    tag_targets: torch.Tensor,
    valence_arousal_pred: Optional[torch.Tensor] = None,
    valence_arousal_true: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """L = L_tags + alpha*||v - v_hat||^2 + beta*||a - a_hat||^2

    Both loss terms are masked against NaN sentinel rows (see
    datasets.MusicContextExample / build_real_examples): a DEAM-sourced
    example has no tags (tags_multihot is all-NaN), an FMA-sourced example
    has no emotion labels (valence_arousal is NaN). Without this masking,
    a single NaN row poisons the whole-batch mean for BOTH loss terms
    (torch's default mean reduction propagates one NaN to every element),
    silently producing a NaN loss/gradient for the entire batch the moment
    real FMA+DEAM data is combined — this masking is required correctness,
    not an optional refinement, once mixed-source batches are in play.
    """
    tag_valid = ~torch.isnan(tag_targets).any(dim=1)
    if tag_valid.any():
        bce = nn.functional.binary_cross_entropy_with_logits(tag_logits[tag_valid], tag_targets[tag_valid])
    else:
        bce = torch.zeros((), device=tag_logits.device, dtype=tag_logits.dtype)
    loss = bce
    parts = {"tag_loss": bce.item()}

    if valence_arousal_pred is not None and valence_arousal_true is not None:
        # mask rows without emotion labels (NaN sentinel) so tracks lacking
        # DEAM annotations don't corrupt the regression loss
        valid = ~torch.isnan(valence_arousal_true).any(dim=1)
        if valid.any():
            v_pred, a_pred = valence_arousal_pred[valid, 0], valence_arousal_pred[valid, 1]
            v_true, a_true = valence_arousal_true[valid, 0], valence_arousal_true[valid, 1]
            v_loss = torch.mean((v_true - v_pred) ** 2)
            a_loss = torch.mean((a_true - a_pred) ** 2)
            loss = loss + alpha * v_loss + beta * a_loss
            parts["valence_loss"] = v_loss.item()
            parts["arousal_loss"] = a_loss.item()

    parts["total_loss"] = loss.item()
    return loss, parts
