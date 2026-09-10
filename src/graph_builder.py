"""
src/graph_builder.py

Section 3, step 3 "Graph construction":
  - Chord-transition graph: nodes = unique chords; edges = observed
    transitions weighted by count.
  - Segment graph: nodes = time segments; edges = temporal adjacency +
    cosine similarity of MFCC/chroma > tau.

Both builders return a lightweight, framework-agnostic `MusicGraph` (plain
numpy arrays) plus a `to_pyg_data()` convenience method that lazily imports
torch_geometric only when you actually want a `Data` object for the GNN
(src/gnn_model.py). This keeps graph construction testable without a GPU
stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.audio_features import SegmentFeatures, dominant_chord_labels, build_chord_templates


@dataclass
class MusicGraph:
    """node_features: [N, F]; edge_index: [2, E]; edge_weight: [E]"""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    node_labels: Optional[List[str]] = field(default=None)  # human-readable labels (chord names, "seg_i")
    graph_type: str = "segment"  # "segment" | "chord"

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to_pyg_data(self, y: Optional[np.ndarray] = None):
        """Convert to a torch_geometric.data.Data object (lazy import)."""
        import torch
        from torch_geometric.data import Data

        data = Data(
            x=torch.tensor(self.node_features, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_weight, dtype=torch.float32).unsqueeze(-1),
        )
        if y is not None:
            data.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return data

    def to_json_dict(self) -> dict:
        """Serializable form for saving `data/processed/*.json` graph samples
        (submission requirement: "at least 20 example .pt / .json graphs")."""
        return {
            "graph_type": self.graph_type,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_features": self.node_features.tolist(),
            "edge_index": self.edge_index.tolist(),
            "edge_weight": self.edge_weight.tolist(),
            "node_labels": self.node_labels,
        }


# -----------------------------------------------------------------------------
# Chord-transition graph
# -----------------------------------------------------------------------------
def build_chord_transition_graph(segments: List[SegmentFeatures]) -> MusicGraph:
    """
    Nodes = unique chords observed in the track (subset of the 24 major/minor
    triads). Edges = observed chord_i -> chord_{i+1} transitions, weighted by
    how many times that transition occurs.
    """
    chord_labels = dominant_chord_labels(segments)
    templates = build_chord_templates()

    unique_chords = sorted(set(chord_labels))
    chord_to_node = {c: i for i, c in enumerate(unique_chords)}

    # Node features = the chord's chroma template (a stable, chord-identity feature)
    node_features = templates[unique_chords]

    transition_counts = {}
    for a, b in zip(chord_labels[:-1], chord_labels[1:]):
        key = (chord_to_node[a], chord_to_node[b])
        transition_counts[key] = transition_counts.get(key, 0) + 1

    if transition_counts:
        edge_index = np.array(list(transition_counts.keys()), dtype=np.int64).T  # [2, E]
        edge_weight = np.array(list(transition_counts.values()), dtype=np.float32)
        edge_weight = edge_weight / edge_weight.max()  # normalize counts to [0, 1]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    from src.audio_features import CHORD_NAMES

    node_labels = [CHORD_NAMES[c] for c in unique_chords]

    return MusicGraph(
        node_features=node_features.astype(np.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_labels=node_labels,
        graph_type="chord",
    )


# -----------------------------------------------------------------------------
# Segment-similarity graph
# -----------------------------------------------------------------------------
def build_segment_graph(
    segments: List[SegmentFeatures],
    similarity_threshold: float = 0.85,
    feature: str = "mfcc",
    add_temporal_edges: bool = True,
) -> MusicGraph:
    """
    Nodes = time segments (in temporal order). Edges =
      (a) temporal adjacency: segment_i -- segment_{i+1}
      (b) similarity edges: cosine_sim(feat_i, feat_j) > tau, for non-adjacent
          pairs too (captures repeated sections / verse-chorus structure).
    """
    n = len(segments)
    feats = np.stack([getattr(s, feature) for s in segments], axis=0).astype(np.float32)  # [N, F]

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    if add_temporal_edges:
        for i in range(n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
            weights.append(1.0)
            weights.append(1.0)

    if n > 1:
        norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim = norm @ norm.T
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if abs(i - j) == 1:
                    continue  # already added as a temporal edge above
                if sim[i, j] > similarity_threshold:
                    edges.append((i, j))
                    weights.append(float(sim[i, j]))

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_weight = np.array(weights, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    node_labels = [f"seg_{i}_{s.start_sec:.1f}-{s.end_sec:.1f}s" for i, s in enumerate(segments)]

    return MusicGraph(
        node_features=feats,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_labels=node_labels,
        graph_type="segment",
    )


def build_graph_for_track(
    segments: List[SegmentFeatures],
    graph_type: str = "segment",
    similarity_threshold: float = 0.85,
) -> MusicGraph:
    """Dispatch helper used by datasets.py so callers don't need to import
    both builder functions."""
    if graph_type == "chord":
        return build_chord_transition_graph(segments)
    elif graph_type == "segment":
        return build_segment_graph(segments, similarity_threshold=similarity_threshold)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
