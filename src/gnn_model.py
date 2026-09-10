"""
src/gnn_model.py

Task 2 (Medium): GNN on Music Structure Graphs.

    GraphSAGE update:
        h_i^{l+1} = sigma( W^{l} . CONCAT( h_i^{l}, MEAN_{j in N(i)} h_j^{l} ) )

    Graph readout (mean pooling):
        g = 1/|V| * sum_{i in V} h_i^{L},   y_hat = sigmoid(W g + b)

Also exposes a GAT variant (config.task2_gnn.encoder = "gat") since the spec
explicitly allows "GraphSAGE or GAT" for Task 2, and the same GNNEncoder is
reused as the structural tower inside Task 3's fusion model.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool, global_mean_pool, global_add_pool


class GNNEncoder(nn.Module):
    """Stacked GraphSAGE or GAT layers -> node embeddings h_i^{L}.
    Pooling into a single graph embedding `g` is exposed separately (`pool`)
    so Task 3's cross-attention fusion can use the *node-level* states too.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        encoder: Literal["graphsage", "gat"] = "graphsage",
        gat_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 1
        self.encoder_type = encoder
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for l in range(num_layers):
            in_dim, out_dim = dims[l], dims[l + 1]
            if encoder == "graphsage":
                self.convs.append(SAGEConv(in_dim, out_dim))
            elif encoder == "gat":
                heads = gat_heads if l < num_layers - 1 else 1
                concat = l < num_layers - 1
                if concat:
                    assert out_dim % heads == 0, (
                        f"GAT with concat=True needs out_dim ({out_dim}) divisible by "
                        f"gat_heads ({heads}); adjust hidden_channels or gat_heads in config."
                    )
                self.convs.append(GATConv(in_dim, out_dim // heads if concat else out_dim, heads=heads, concat=concat))
            else:
                raise ValueError(f"Unknown encoder type: {encoder}")
            self.norms.append(nn.LayerNorm(out_dim))

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for l, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if l < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [N_total_nodes_in_batch, out_channels]

    @staticmethod
    def pool(h: torch.Tensor, batch: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Graph-level readout g. `batch` is the PyG batch-assignment vector
        mapping each node to its graph index within a mini-batch."""
        if mode == "mean":
            return global_mean_pool(h, batch)
        elif mode == "max":
            return global_max_pool(h, batch)
        elif mode == "sum":
            return global_add_pool(h, batch)
        raise ValueError(f"Unknown pooling mode: {mode}")


class GNNGenreClassifier(nn.Module):
    """
    Full Task-2 model: GNNEncoder + mean-pool readout + linear head, trained
    with multi-label BCE (genre / top-tags) as in Algorithm 2.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        encoder: str = "graphsage",
        gat_heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.gnn = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            encoder=encoder,
            gat_heads=gat_heads,
            dropout=dropout,
        )
        self.pooling = pooling
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.gnn(x, edge_index)
        g = self.gnn.pool(h, batch, mode=self.pooling)
        logits = self.head(g)
        return logits

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """Returns (node_embeddings, graph_embedding) — used by Task 3/4 and
        by the graph_coherence_score / t-SNE analyses in evaluate.py."""
        h = self.gnn(x, edge_index)
        g = self.gnn.pool(h, batch, mode=self.pooling)
        return h, g


# -----------------------------------------------------------------------------
# Baseline B2: CNN on mel-spectrogram (no graph, no text) — Section 8
# -----------------------------------------------------------------------------
class MelSpecCNN(nn.Module):
    """A compact CNN baseline over log-mel spectrograms, used as B2 in the
    required baseline comparison (Section 8) and Table 3's "CNN mel-spec" row."""

    def __init__(self, num_classes: int, n_mels: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 1, n_mels, T]
        h = self.features(mel)
        h = h.flatten(1)
        return self.classifier(h)
