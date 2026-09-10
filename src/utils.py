"""
src/utils.py

Cross-cutting utilities shared by every task: reproducibility, config loading,
checkpointing, and the evaluation metrics defined in Section 6 of the project
spec (Macro/Micro-F1, AUC-PR, MAE, R^2, retrieval R@K, graph coherence score).

Kept dependency-light on purpose: only numpy / sklearn / yaml / torch (torch is
imported lazily inside functions that need it) so this module can also be unit
tested in environments without a GPU stack installed.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch (+ CUDA) RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic-ish; full determinism with GNN scatter ops is not
        # guaranteed by cuDNN, so we don't force torch.use_deterministic_algorithms.
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preferred: str = "cuda") -> "Any":
    """Return a torch.device, falling back to CPU if CUDA was requested but
    unavailable (e.g. running the pipeline smoke-test on a laptop)."""
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: str = "config.yaml", overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load the YAML config and apply dotted-path CLI overrides such as
    ``training.epochs=5`` or ``data.mode=real``."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key.path=value")
        key_path, raw_value = item.split("=", 1)
        value = _coerce(raw_value)
        _set_by_dotted_path(cfg, key_path, value)
    return cfg


def _coerce(raw: str) -> Any:
    """Turn a CLI string into int/float/bool/str as appropriate."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def _set_by_dotted_path(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
def save_checkpoint(model, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path: str, map_location: str = "cpu") -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def dump_metrics(metrics: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(metrics)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# -----------------------------------------------------------------------------
# Metrics — Section 6 of the spec
# -----------------------------------------------------------------------------
@dataclass
class TagMetrics:
    macro_f1: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    auc_pr_mean: float
    per_tag_f1: List[float]
    per_tag_auc_pr: List[float]


def compute_tag_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> TagMetrics:
    """
    y_true: [N, K] binary ground-truth multi-label matrix
    y_prob: [N, K] predicted probabilities in [0, 1]

    Implements Prec_k / Rec_k / F1_k, Macro-F1, Micro-F1 and mean AUC-PR
    exactly as defined in Section 6 of the spec.
    """
    y_pred = (y_prob >= threshold).astype(int)

    macro_p, macro_r, per_tag_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = float(np.mean(per_tag_f1))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    per_tag_auc_pr = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0 or y_true[:, k].sum() == len(y_true):
            # AUC-PR undefined for a tag with no positive (or no negative) examples
            per_tag_auc_pr.append(float("nan"))
        else:
            per_tag_auc_pr.append(float(average_precision_score(y_true[:, k], y_prob[:, k])))

    auc_pr_mean = float(np.nanmean(per_tag_auc_pr)) if len(per_tag_auc_pr) else 0.0

    return TagMetrics(
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        macro_precision=float(np.mean(macro_p)),
        macro_recall=float(np.mean(macro_r)),
        auc_pr_mean=auc_pr_mean,
        per_tag_f1=[float(x) for x in per_tag_f1],
        per_tag_auc_pr=per_tag_auc_pr,
    )


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    micro_f1: float


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Single-label multi-class metrics (e.g. genre classification on
    GTZAN/FMA), as opposed to the multi-label `compute_tag_metrics` above.
    y_true, y_pred: 1-D arrays of integer class indices."""
    from sklearn.metrics import accuracy_score

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        micro_f1=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    )


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE and R^2 for valence/arousal regression (DEAM), as in Section 6."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "r2": r2}


@dataclass
class EmotionStats:
    """Per-dimension mean/std fit on the TRAIN split only, used to z-score
    valence/arousal before they enter the regression loss (see fix note in
    fusion_model.fusion_multitask_loss / train.py train_task3).

    Why this exists: raw valence/arousal live on a 1-9 scale (DEAM's native
    range). Squared error against an untrained linear head on that raw scale
    is ~30 on average, which at alpha=beta=0.5 completely swamps a tag BCE
    loss of ~0.6 — the fusion model's tag head never gets a usable gradient
    signal before early stopping kicks in. Z-scoring the targets (and
    de-normalizing predictions before computing MAE/R^2) puts both loss terms
    on comparable scales without changing what MAE/R^2 mean to a reader.
    """

    v_mean: float
    v_std: float
    a_mean: float
    a_std: float

    def normalize(self, va: np.ndarray) -> np.ndarray:
        out = va.copy()
        out[:, 0] = (out[:, 0] - self.v_mean) / self.v_std
        out[:, 1] = (out[:, 1] - self.a_mean) / self.a_std
        return out

    def denormalize(self, va: np.ndarray) -> np.ndarray:
        out = va.copy()
        out[:, 0] = out[:, 0] * self.v_std + self.v_mean
        out[:, 1] = out[:, 1] * self.a_std + self.a_mean
        return out


def fit_emotion_stats(valence: List[Optional[float]], arousal: List[Optional[float]]) -> EmotionStats:
    """Fit z-score stats from the TRAIN split's valence/arousal, skipping
    tracks without emotion labels (None)."""
    v = np.array([x for x in valence if x is not None], dtype=np.float64)
    a = np.array([x for x in arousal if x is not None], dtype=np.float64)
    v_mean, v_std = (float(v.mean()), float(v.std() + 1e-6)) if len(v) else (0.0, 1.0)
    a_mean, a_std = (float(a.mean()), float(a.std() + 1e-6)) if len(a) else (0.0, 1.0)
    return EmotionStats(v_mean=v_mean, v_std=v_std, a_mean=a_mean, a_std=a_std)


def retrieval_recall_at_k(similarity: np.ndarray, ks: List[int] = (1, 5, 10)) -> Dict[str, float]:
    """
    Caption->Audio (or Audio->Caption) Recall@K for a contrastive retrieval task.

    similarity: [N, N] matrix where similarity[i, j] = sim(query_i, candidate_j)
    and the correct match for query i is candidate i (paired, diagonal ground truth,
    matching the InfoNCE setup in Task 4 / Algorithm 4).
    """
    n = similarity.shape[0]
    ranks = np.zeros(n, dtype=int)
    order = np.argsort(-similarity, axis=1)  # descending similarity
    for i in range(n):
        # position (0-indexed) of the true match i within the sorted candidates
        ranks[i] = int(np.where(order[i] == i)[0][0])

    out = {}
    for k in ks:
        out[f"R@{k}"] = float(np.mean(ranks < k))
    out["median_rank"] = float(np.median(ranks) + 1)
    return out


def graph_coherence_score(node_embeddings: np.ndarray, edges: np.ndarray, tau: float = 0.5) -> float:
    """
    Optional analysis metric (Section 6): fraction of edges (i, j) whose endpoint
    embeddings have cosine similarity above tau — a proxy for whether the GNN's
    learned representations respect the graph's structural (e.g. chord-repeat)
    connectivity.

    node_embeddings: [N, d]
    edges: [E, 2] array of (i, j) node index pairs
    """
    if len(edges) == 0:
        return 0.0
    norm = node_embeddings / (np.linalg.norm(node_embeddings, axis=1, keepdims=True) + 1e-12)
    src, dst = edges[:, 0], edges[:, 1]
    cos_sim = np.sum(norm[src] * norm[dst], axis=1)
    return float(np.mean(cos_sim > tau))


def majority_class_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """B1 baseline: predict the training-set marginal frequency for every tag,
    for every test example (a constant, non-learned predictor)."""
    tag_freq = y_train.mean(axis=0, keepdims=True)  # [1, K]
    return np.repeat(tag_freq, n_test, axis=0)
