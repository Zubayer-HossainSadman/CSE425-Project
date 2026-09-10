"""
src/contrastive.py

Task 4 (Advanced): Cross-Modal MusicCaps Alignment.

    InfoNCE loss for paired (graph, caption) (g_i, t_i):
        L_NCE = -log[ exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau) ]
        sim(u, v) = u^T v / (||u|| ||v||)

    Retrieval metrics: Caption->Audio R@1/R@5/R@10, Audio->Caption R@K
    (see src/utils.retrieval_recall_at_k for the metric implementation).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import BertTextEncoder
from src.gnn_model import GNNEncoder


class ContrastiveGNNBert(nn.Module):
    """
    Dual encoder: a GNN tower for audio-structure graphs and a BERT tower for
    MusicCaps captions, each projected into a shared embedding space and
    L2-normalized, trained with symmetric InfoNCE (Algorithm 4).
    """

    def __init__(
        self,
        gnn_in_channels: int,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
        gnn_type: str = "graphsage",
        bert_model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 256,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn = GNNEncoder(
            in_channels=gnn_in_channels,
            hidden_channels=gnn_hidden,
            out_channels=gnn_hidden,
            num_layers=gnn_layers,
            encoder=gnn_type,
            dropout=dropout,
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(gnn_hidden, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.text_encoder = BertTextEncoder(model_name=bert_model_name)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # log-parameterized, learnable temperature (kept close to the CLIP recipe);
        # initialized from config's `temperature` via its log.
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def encode_audio(self, graph_x: torch.Tensor, graph_edge_index: torch.Tensor, graph_batch: torch.Tensor) -> torch.Tensor:
        h = self.gnn(graph_x, graph_edge_index)
        g = self.gnn.pool(h, graph_batch, mode="mean")
        g = self.audio_proj(g)
        return F.normalize(g, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        t, _ = self.text_encoder(input_ids, attention_mask)
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_emb = self.encode_audio(graph_x, graph_edge_index, graph_batch)
        text_emb = self.encode_text(input_ids, attention_mask)
        temperature = self.log_temperature.exp().clamp(min=1e-3, max=100.0)
        return audio_emb, text_emb, temperature


def info_nce_loss(audio_emb: torch.Tensor, text_emb: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE over an in-batch similarity matrix S_ij = a_i . t_j / tau
    (Algorithm 4). Embeddings are assumed already L2-normalized so the dot
    product equals cosine similarity.
    """
    logits = audio_emb @ text_emb.t() / temperature  # [B, B]
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_a2t = F.cross_entropy(logits, targets)         # audio -> caption
    loss_t2a = F.cross_entropy(logits.t(), targets)      # caption -> audio
    return 0.5 * (loss_a2t + loss_t2a)


@torch.no_grad()
def build_similarity_matrix(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix for retrieval evaluation (rows=audio queries,
    cols=text candidates, or vice-versa — symmetric since both are normalized)."""
    return audio_emb @ text_emb.t()


def zero_shot_tag_prediction(text_emb: torch.Tensor, tag_text_emb: torch.Tensor) -> torch.Tensor:
    """
    Deliverable: "Zero-shot tag prediction from captions vs. Task 3 supervised
    model". Given caption embeddings and embeddings of each candidate tag's
    natural-language description (e.g. "a melancholic jazz piece"), score each
    tag by cosine similarity — no supervised tag classifier involved.

    text_emb:     [N, d]   normalized caption/audio embeddings
    tag_text_emb: [K, d]   normalized embeddings of tag prompt strings
    returns:      [N, K]   similarity scores usable as pseudo-probabilities
    """
    return text_emb @ tag_text_emb.t()
