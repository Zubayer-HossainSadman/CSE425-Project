"""
train.py — unified training entry point for Tasks 1-4.

Usage:
    python train.py --task 1 --config config.yaml
    python train.py --task 2 --config config.yaml --override training.epochs=10
    python train.py --task 3 --config config.yaml --override task3_fusion.fusion_type=concat
    python train.py --task 4 --config config.yaml
    python train.py --task all --config config.yaml     # runs 1 -> 2 -> 3 -> 4 in sequence

Run with `data.mode: synthetic` (default in config.yaml) to smoke-test the
whole pipeline without downloading FMA/MagnaTagATune/MusicCaps/DEAM. Switch to
`data.mode: real` once those are in data/raw/ (see Table 1 in the spec / the
README for links) and the paths in config.yaml point at them.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.bert_encoder import BertTagClassifier
from src.contrastive import ContrastiveGNNBert, info_nce_loss, build_similarity_matrix, zero_shot_tag_prediction
from src.datasets import (
    MusicContextDataset,
    build_real_examples,
    build_synthetic_examples,
    filter_examples,
    has_valid_genre,
    has_valid_tags,
    make_collate_fn,
)
from src.fusion_model import GNNBertFusionModel, fusion_multitask_loss
from src.gnn_model import GNNGenreClassifier, MelSpecCNN
from src.synthetic_data import GENRE_NAMES, synthetic_log_mel
from src.utils import (
    compute_classification_metrics,
    compute_emotion_metrics,
    compute_tag_metrics,
    dump_metrics,
    fit_emotion_stats,
    get_device,
    majority_class_baseline,
    retrieval_recall_at_k,
    save_checkpoint,
    set_seed,
    load_config,
)


# -----------------------------------------------------------------------------
# Data loading (shared across tasks)
# -----------------------------------------------------------------------------
def get_splits(cfg: Dict, graph_type: str = "segment", require: Optional[str] = None) -> Dict:
    """
    require: None (Task 3/4 — use every example, real or synthetic) |
             "tags" (Task 1 — drop examples with no tag labels, e.g. DEAM) |
             "genre" (Task 2 — drop examples with no genre label, e.g. DEAM)
    Synthetic examples always have both, so `require` is a no-op there.
    """
    if cfg["data"]["mode"] == "synthetic":
        syn_cfg = cfg["data"]["synthetic"]
        splits = build_synthetic_examples(
            n_tracks=syn_cfg["n_tracks"],
            n_genres=len(GENRE_NAMES),
            top_k_tags=cfg["data"]["synthetic"]["n_tags"],
            graph_type=graph_type,
            seed=cfg["project"]["seed"],
        )
    elif cfg["data"]["mode"] == "real":
        splits = build_real_examples(cfg, graph_type=graph_type)
    else:
        raise ValueError(f"Unknown data.mode: {cfg['data']['mode']!r} (expected 'synthetic' or 'real')")

    if require == "tags":
        for split in ("train", "val", "test"):
            splits[split] = filter_examples(splits[split], has_valid_tags)
    elif require == "genre":
        for split in ("train", "val", "test"):
            splits[split] = filter_examples(splits[split], has_valid_genre)
    return splits


def build_loaders(splits: Dict, tokenizer, batch_size: int, max_length: int, num_workers: int = 0):
    for split in ("train", "val", "test"):
        if len(splits[split]) == 0:
            raise RuntimeError(
                f"The '{split}' split has 0 examples after loading/filtering — training/evaluation "
                f"cannot proceed. Most likely causes: (1) the audio archive (e.g. fma_medium.zip) "
                f"hasn't finished extracting yet, so most track_ids don't have a matching file on disk "
                f"yet; (2) the metadata/annotations path in config.yaml doesn't point at your actual "
                f"extracted folder; (3) for Task 1/2 specifically, every example in this split lacked "
                f"the required label (tags/genre) — check the '[build_real_examples] FMA: ... DEAM: ...' "
                f"summary line printed just above this for how many tracks were actually found."
            )
    collate = make_collate_fn(tokenizer, max_length=max_length)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = MusicContextDataset(splits[split])
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=collate, num_workers=num_workers, drop_last=False,
        )
    return loaders


def infer_graph_in_channels(splits: Dict) -> int:
    return int(splits["train"][0].graph.node_features.shape[1])


# -----------------------------------------------------------------------------
# Task 1: BERT baseline
# -----------------------------------------------------------------------------
def train_task1(cfg: Dict, device: torch.device) -> Dict:
    print("\n=== Task 1: BERT Multi-Label Tag Classifier ===")
    splits = get_splits(cfg, graph_type="segment", require="tags")
    tag_vocab = splits["tag_vocab"]

    model_cfg = cfg["task1_bert"]
    model = BertTagClassifier(
        num_tags=len(tag_vocab),
        model_name=model_cfg["model_name"],
        max_length=cfg["preprocessing"]["max_text_length"],
        freeze_bert=model_cfg["freeze_bert"],
        hidden_dropout=model_cfg["hidden_dropout"],
    ).to(device)

    loaders = build_loaders(
        splits, model.text_encoder.tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["preprocessing"]["max_text_length"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": [], "val_auc_pr": []}
    best_auc_pr, best_state, patience = -1.0, None, 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["tags"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(loaders["train"]))

        val_metrics = evaluate_task1(model, loaders["val"], device)
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_metrics.macro_f1)
        history["val_micro_f1"].append(val_metrics.micro_f1)
        history["val_auc_pr"].append(val_metrics.auc_pr_mean)
        print(f"[Task1] epoch {epoch+1}/{cfg['training']['epochs']} "
              f"loss={epoch_loss:.4f} val_macro_f1={val_metrics.macro_f1:.4f} val_micro_f1={val_metrics.micro_f1:.4f} "
              f"val_auc_pr={val_metrics.auc_pr_mean:.4f}")

        # Early-stop / checkpoint-select on AUC-PR, not macro-F1: macro-F1
        # thresholds probabilities at 0.5, so it reads exactly 0.0000 for
        # every epoch until the model's calibration crosses that threshold
        # (observed: 4-5 epochs before any tag's prob crosses 0.5). AUC-PR is
        # threshold-free and moves from epoch 1, so it won't let patience run
        # out before the model has had a fair chance to calibrate.
        if val_metrics.auc_pr_mean > best_auc_pr:
            best_auc_pr, best_state, patience = val_metrics.auc_pr_mean, model.state_dict(), 0
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("[Task1] early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_task1(model, loaders["test"], device)
    print(f"[Task1] TEST macro_f1={test_metrics.macro_f1:.4f} micro_f1={test_metrics.micro_f1:.4f} "
          f"auc_pr={test_metrics.auc_pr_mean:.4f}")

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    save_checkpoint(model, os.path.join(cfg["training"]["checkpoint_dir"], "task1_bert.pt"))
    _plot_f1_curve(history, "results/plots/task1_f1_curve.png", title="Task 1: BERT Tag Classifier")
    _dump_example_predictions(model, loaders["test"], tag_vocab, device, "results/task1_example_predictions.json")

    results = {
        "task1_bert": {
            "test_macro_f1": test_metrics.macro_f1,
            "test_micro_f1": test_metrics.micro_f1,
            "test_auc_pr": test_metrics.auc_pr_mean,
            "history": history,
        }
    }
    dump_metrics(results, os.path.join(cfg["project"]["output_dir"], "metrics.json"))
    return results


@torch.no_grad()
def evaluate_task1(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    for batch in loader:
        probs = model.predict_proba(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_probs.append(probs.cpu().numpy())
        all_true.append(batch["tags"].numpy())
    return compute_tag_metrics(np.concatenate(all_true), np.concatenate(all_probs))


@torch.no_grad()
def _dump_example_predictions(model, loader, tag_vocab: List[str], device, path: str, n: int = 5):
    import json

    model.eval()
    batch = next(iter(loader))
    probs = model.predict_proba(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu().numpy()
    examples = []
    for i in range(min(n, len(batch["track_ids"]))):
        top_idx = np.argsort(-probs[i])[:5]
        true_idx = np.where(batch["tags"][i].numpy() > 0.5)[0]
        examples.append(
            {
                "track_id": batch["track_ids"][i],
                "predicted_top5_tags": [tag_vocab[j] for j in top_idx],
                "true_tags": [tag_vocab[j] for j in true_idx],
            }
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"[Task1] wrote {len(examples)} example predictions -> {path}")


# -----------------------------------------------------------------------------
# Task 2: GNN on structure graphs (+ CNN baseline B2, + majority baseline B1)
# -----------------------------------------------------------------------------
def train_task2(cfg: Dict, device: torch.device) -> Dict:
    print("\n=== Task 2: GNN on Music Structure Graphs ===")
    splits = get_splits(cfg, graph_type="segment", require="genre")
    genre_vocab = splits["genre_vocab"]
    in_channels = infer_graph_in_channels(splits)

    gnn_cfg = cfg["task2_gnn"]
    model = GNNGenreClassifier(
        in_channels=in_channels,
        num_classes=len(genre_vocab),
        hidden_channels=gnn_cfg["hidden_channels"],
        out_channels=gnn_cfg["out_channels"],
        num_layers=gnn_cfg["num_layers"],
        encoder=gnn_cfg["encoder"],
        gat_heads=gnn_cfg["gat_heads"],
        dropout=gnn_cfg["dropout"],
        pooling=gnn_cfg["pooling"],
    ).to(device)

    # Text isn't used by Task 2, but build_loaders always tokenizes `text` for
    # a uniform collate_fn; a tiny throwaway tokenizer keeps this cheap.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["preprocessing"]["bert_model_name"])
    loaders = build_loaders(splits, tokenizer, cfg["training"]["batch_size"], cfg["preprocessing"]["max_text_length"])

    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_cfg["lr"], weight_decay=gnn_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_macro_f1": [], "val_acc": []}
    best_macro_f1, best_state, patience = -1.0, None, 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            logits = model(batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device))
            loss = criterion(logits, batch["genre_idx"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(loaders["train"]))

        val_metrics = evaluate_task2_gnn(model, loaders["val"], device)
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_metrics.macro_f1)
        history["val_acc"].append(val_metrics.accuracy)
        print(f"[Task2/GNN] epoch {epoch+1}/{cfg['training']['epochs']} "
              f"loss={epoch_loss:.4f} val_acc={val_metrics.accuracy:.4f} val_macro_f1={val_metrics.macro_f1:.4f}")

        if val_metrics.macro_f1 > best_macro_f1:
            best_macro_f1, best_state, patience = val_metrics.macro_f1, model.state_dict(), 0
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("[Task2/GNN] early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_task2_gnn(model, loaders["test"], device)
    print(f"[Task2/GNN] TEST acc={test_metrics.accuracy:.4f} macro_f1={test_metrics.macro_f1:.4f}")

    save_checkpoint(model, os.path.join(cfg["training"]["checkpoint_dir"], "task2_gnn.pt"))
    _plot_f1_curve(history, "results/plots/task2_gnn_f1_curve.png", title="Task 2: GNN Genre Classifier", f1_key="val_macro_f1")

    # --- B1: majority-class baseline ---
    train_genre_onehot = np.eye(len(genre_vocab))[[e.genre_idx for e in splits["train"]]]
    test_genre_true = np.array([e.genre_idx for e in splits["test"]])
    maj_probs = majority_class_baseline(train_genre_onehot, len(test_genre_true))
    maj_pred = np.argmax(maj_probs, axis=1)
    b1_metrics = compute_classification_metrics(test_genre_true, maj_pred)
    print(f"[Task2/B1-Majority] TEST acc={b1_metrics.accuracy:.4f} macro_f1={b1_metrics.macro_f1:.4f}")

    # --- B2: CNN on synthetic mel-spectrogram baseline ---
    b2_metrics = train_cnn_baseline(cfg, splits, genre_vocab, device)

    results = {
        "task2_gnn": {
            "test_accuracy": test_metrics.accuracy,
            "test_macro_f1": test_metrics.macro_f1,
            "history": history,
        },
        "baseline_b1_majority": {"test_accuracy": b1_metrics.accuracy, "test_macro_f1": b1_metrics.macro_f1},
        "baseline_b2_cnn_melspec": b2_metrics,
    }
    dump_metrics(results, os.path.join(cfg["project"]["output_dir"], "metrics.json"))
    return results


@torch.no_grad()
def evaluate_task2_gnn(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for batch in loader:
        logits = model(batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device))
        all_pred.append(logits.argmax(dim=-1).cpu().numpy())
        all_true.append(batch["genre_idx"].numpy())
    return compute_classification_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def train_cnn_baseline(cfg: Dict, splits: Dict, genre_vocab: List[str], device: torch.device) -> Dict:
    """B2: CNN on (synthetic) mel-spectrograms — see Section 8 and Table 3."""
    print("[Task2/B2-CNN] training mel-spectrogram CNN baseline...")

    def to_tensor_batch(examples, idx_start, idx_end):
        mels, genres = [], []
        for i in range(idx_start, min(idx_end, len(examples))):
            ex = examples[i]
            track_num = int(ex.track_id.split("_")[-1])
            mels.append(synthetic_log_mel(ex.genre_idx, track_num))
            genres.append(ex.genre_idx)
        x = torch.tensor(np.stack(mels), dtype=torch.float32).unsqueeze(1)  # [B, 1, n_mels, T]
        y = torch.tensor(genres, dtype=torch.long)
        return x, y

    model = MelSpecCNN(num_classes=len(genre_vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    batch_size = cfg["training"]["batch_size"]

    train_ex, test_ex = splits["train"], splits["test"]
    for epoch in range(min(10, cfg["training"]["epochs"])):  # CNN baseline converges fast; cap epochs
        model.train()
        perm = np.random.permutation(len(train_ex))
        for start in range(0, len(train_ex), batch_size):
            idx = perm[start:start + batch_size]
            x, y = to_tensor_batch([train_ex[i] for i in idx], 0, len(idx))
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x, y = to_tensor_batch(test_ex, 0, len(test_ex))
        preds = model(x.to(device)).argmax(dim=-1).cpu().numpy()
    metrics = compute_classification_metrics(y.numpy(), preds)
    print(f"[Task2/B2-CNN] TEST acc={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}")
    return {"test_accuracy": metrics.accuracy, "test_macro_f1": metrics.macro_f1}


# -----------------------------------------------------------------------------
# Task 3: GNN-BERT fusion (+ ablations)
# -----------------------------------------------------------------------------
def train_task3(cfg: Dict, device: torch.device, ablation: str = "cross_attention") -> Dict:
    """
    ablation in {"cross_attention", "concat", "bert_only", "gnn_only"} — the
    four variants required by the Task-3 ablation deliverable.
    """
    print(f"\n=== Task 3: GNN-BERT Fusion (ablation={ablation}) ===")
    splits = get_splits(cfg, graph_type="segment")
    tag_vocab = splits["tag_vocab"]
    in_channels = infer_graph_in_channels(splits)
    fusion_cfg = cfg["task3_fusion"]

    # Fit valence/arousal normalization stats from the TRAIN split only (never
    # val/test — that would leak label distribution info). Fixes a real bug:
    # raw 1-9 scale targets against an untrained regression head produce a
    # squared error ~30, which at alpha=beta=0.5 swamps the ~0.6 tag BCE loss
    # and starves the tag head of gradient signal. See utils.EmotionStats.
    emotion_stats = fit_emotion_stats(
        [e.valence for e in splits["train"]], [e.arousal for e in splits["train"]]
    )

    if ablation == "bert_only":
        model = BertTagClassifier(num_tags=len(tag_vocab), model_name=cfg["preprocessing"]["bert_model_name"]).to(device)
    elif ablation == "gnn_only":
        model = GNNGenreClassifier(
            in_channels=in_channels, num_classes=len(tag_vocab),
            hidden_channels=64, out_channels=64, num_layers=3, encoder="graphsage",
        ).to(device)
    else:  # "concat" or "cross_attention"
        model = GNNBertFusionModel(
            num_tags=len(tag_vocab),
            gnn_in_channels=in_channels,
            gnn_hidden=fusion_cfg["gnn_out_dim"],
            gnn_out=fusion_cfg["gnn_out_dim"],
            bert_model_name=cfg["preprocessing"]["bert_model_name"],
            fusion_type=ablation,
            fusion_hidden_dim=fusion_cfg["fusion_hidden_dim"],
            predict_emotion=fusion_cfg["predict_emotion"],
        ).to(device)

    tokenizer_source = model.text_encoder.tokenizer if hasattr(model, "text_encoder") else None
    if tokenizer_source is None:
        from transformers import AutoTokenizer
        tokenizer_source = AutoTokenizer.from_pretrained(cfg["preprocessing"]["bert_model_name"])

    loaders = build_loaders(splits, tokenizer_source, cfg["training"]["batch_size"], cfg["preprocessing"]["max_text_length"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=fusion_cfg["lr"], weight_decay=fusion_cfg["weight_decay"])

    history = {"train_loss": [], "val_macro_f1": [], "val_auc_pr": []}
    best_auc_pr, best_state, patience = -1.0, None, 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss = _task3_forward_loss(model, batch, device, ablation, fusion_cfg, emotion_stats)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(loaders["train"]))

        val_metrics, _ = evaluate_task3(model, loaders["val"], device, ablation, emotion_stats=emotion_stats)
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_metrics["tag_macro_f1"])
        history["val_auc_pr"].append(val_metrics["tag_auc_pr"])
        print(f"[Task3/{ablation}] epoch {epoch+1}/{cfg['training']['epochs']} "
              f"loss={epoch_loss:.4f} val_macro_f1={val_metrics['tag_macro_f1']:.4f} val_auc_pr={val_metrics['tag_auc_pr']:.4f}")

        # Same fix as Task 1: select/early-stop on AUC-PR (threshold-free),
        # not macro-F1 (reads a hard 0.0000 until probabilities cross 0.5,
        # which cost concat/cross_attention their entire patience budget
        # before their tag head had calibrated).
        if val_metrics["tag_auc_pr"] > best_auc_pr:
            best_auc_pr, best_state, patience = val_metrics["tag_auc_pr"], model.state_dict(), 0
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print(f"[Task3/{ablation}] early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, embeddings = evaluate_task3(model, loaders["test"], device, ablation, collect_embeddings=True, emotion_stats=emotion_stats)
    print(f"[Task3/{ablation}] TEST macro_f1={test_metrics['tag_macro_f1']:.4f} "
          f"auc_pr={test_metrics.get('tag_auc_pr', float('nan')):.4f}")

    save_checkpoint(model, os.path.join(cfg["training"]["checkpoint_dir"], f"task3_{ablation}.pt"))

    if ablation == "cross_attention":
        _plot_tsne(embeddings, splits["test"], "results/plots/task3_tsne.png")
        _dump_case_studies(model, loaders["test"], tag_vocab, device, "results/task3_case_studies.json")

    return {f"task3_{ablation}": test_metrics, "history": history}


def _task3_forward_loss(model, batch, device, ablation: str, fusion_cfg: Dict, emotion_stats=None):
    tags = batch["tags"].to(device)
    if ablation == "bert_only":
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        return _masked_bce(logits, tags)
    elif ablation == "gnn_only":
        logits = model(batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device))
        return _masked_bce(logits, tags)
    else:
        out = model(
            batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device),
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
        )
        va_true = None
        if fusion_cfg["predict_emotion"]:
            va_raw = batch["valence_arousal"].numpy()
            va_norm = emotion_stats.normalize(va_raw)  # z-score so the regression
            # loss lives on the same scale as the tag BCE loss instead of
            # dominating it (see the comment in train_task3 / utils.EmotionStats)
            va_true = torch.tensor(va_norm, dtype=torch.float32, device=device)
        loss, _ = fusion_multitask_loss(
            out["tag_logits"], tags, out["valence_arousal"], va_true,
            alpha=fusion_cfg["emotion_loss_weight_valence"], beta=fusion_cfg["emotion_loss_weight_arousal"],
        )
        return loss


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Same NaN-row masking as fusion_multitask_loss's tag term, for the
    bert_only/gnn_only ablations that call BCE directly instead of going
    through that helper. Needed the moment real FMA (has tags) + DEAM (all-NaN
    tags) examples can land in the same batch — see build_real_examples."""
    valid = ~torch.isnan(targets).any(dim=1)
    if not valid.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logits[valid], targets[valid])


@torch.no_grad()
def evaluate_task3(model, loader, device, ablation: str, collect_embeddings: bool = False, emotion_stats=None):
    model.eval()
    all_probs, all_true, all_va_pred, all_va_true, all_z = [], [], [], [], []

    for batch in loader:
        tags = batch["tags"].numpy()
        if ablation == "bert_only":
            probs = torch.sigmoid(model(batch["input_ids"].to(device), batch["attention_mask"].to(device))).cpu().numpy()
        elif ablation == "gnn_only":
            probs = torch.sigmoid(
                model(batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device))
            ).cpu().numpy()
        else:
            out = model(
                batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device),
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
            )
            probs = torch.sigmoid(out["tag_logits"]).cpu().numpy()
            if out["valence_arousal"] is not None:
                # model outputs live in normalized (z-scored) space; convert
                # back to the original 1-9 scale so MAE/R^2 are interpretable
                va_pred_norm = out["valence_arousal"].cpu().numpy()
                va_pred_raw = emotion_stats.denormalize(va_pred_norm) if emotion_stats is not None else va_pred_norm
                all_va_pred.append(va_pred_raw)
                all_va_true.append(batch["valence_arousal"].numpy())
            if collect_embeddings:
                all_z.append(out["fused_embedding"].cpu().numpy())

        all_probs.append(probs)
        all_true.append(tags)

    all_true_arr = np.concatenate(all_true)
    all_probs_arr = np.concatenate(all_probs)
    # exclude rows with no tag labels (e.g. DEAM examples) from tag metrics —
    # compute_tag_metrics/sklearn would otherwise choke on NaN ground truth
    tag_valid = ~np.isnan(all_true_arr).any(axis=1)
    tag_metrics = compute_tag_metrics(all_true_arr[tag_valid], all_probs_arr[tag_valid])
    result = {"tag_macro_f1": tag_metrics.macro_f1, "tag_micro_f1": tag_metrics.micro_f1, "tag_auc_pr": tag_metrics.auc_pr_mean}

    if all_va_pred:
        va_pred = np.concatenate(all_va_pred)
        va_true = np.concatenate(all_va_true)
        valid = ~np.isnan(va_true).any(axis=1)
        if valid.any():
            emo = compute_emotion_metrics(va_true[valid], va_pred[valid])
            result["emotion_mae"] = emo["mae"]
            result["emotion_r2"] = emo["r2"]

    embeddings = np.concatenate(all_z) if all_z else None
    return result, embeddings


def run_task3_ablations(cfg: Dict, device: torch.device) -> Dict:
    all_results = {}
    for ablation in ("bert_only", "gnn_only", "concat", "cross_attention"):
        all_results.update(run_single_ablation(cfg, device, ablation))
    dump_metrics(all_results, os.path.join(cfg["project"]["output_dir"], "metrics.json"))
    print("\n[Task3] Ablation summary (test Macro-F1):")
    for k, v in all_results.items():
        if isinstance(v, dict) and "tag_macro_f1" in v:
            print(f"  {k:20s} macro_f1={v['tag_macro_f1']:.4f}")
    return all_results


def run_single_ablation(cfg: Dict, device: torch.device, ablation: str) -> Dict:
    set_seed(cfg["project"]["seed"])  # keep ablations comparable
    return train_task3(cfg, device, ablation=ablation)


# -----------------------------------------------------------------------------
# Task 4: Contrastive dual-encoder (MusicCaps-style alignment)
# -----------------------------------------------------------------------------
def train_task4(cfg: Dict, device: torch.device) -> Dict:
    print("\n=== Task 4: Contrastive GNN-BERT (MusicCaps alignment) ===")
    splits = get_splits(cfg, graph_type="segment")
    tag_vocab = splits["tag_vocab"]
    in_channels = infer_graph_in_channels(splits)
    c_cfg = cfg["task4_contrastive"]

    model = ContrastiveGNNBert(
        gnn_in_channels=in_channels,
        gnn_hidden=64,
        bert_model_name=cfg["preprocessing"]["bert_model_name"],
        embedding_dim=c_cfg["embedding_dim"],
        temperature=c_cfg["temperature"],
    ).to(device)

    loaders = build_loaders(splits, model.text_encoder.tokenizer, cfg["training"]["batch_size"], cfg["preprocessing"]["max_text_length"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c_cfg["lr"], weight_decay=c_cfg["weight_decay"])

    history = {"train_loss": []}
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in loaders["train"]:
            if len(batch["track_ids"]) < 2:
                continue  # InfoNCE needs >=2 examples per batch for in-batch negatives
            optimizer.zero_grad()
            audio_emb, text_emb, temperature = model(
                batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device),
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
            )
            loss = info_nce_loss(audio_emb, text_emb, temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(loaders["train"]))
        history["train_loss"].append(epoch_loss)
        print(f"[Task4] epoch {epoch+1}/{cfg['training']['epochs']} loss={epoch_loss:.4f}")

    save_checkpoint(model, os.path.join(cfg["training"]["checkpoint_dir"], "task4_contrastive.pt"))

    retrieval_metrics, audio_emb, text_emb = evaluate_task4(model, loaders["test"], device, c_cfg["retrieval_ks"])
    print(f"[Task4] TEST retrieval: {retrieval_metrics}")

    _dump_retrieval_examples(splits["test"], audio_emb, text_emb, "results/retrieval_examples/task4_examples.json")

    zs_metrics = zero_shot_tag_eval(model, splits["test"], tag_vocab, device)
    zs_macro_f1 = zs_metrics.macro_f1 if zs_metrics is not None else None
    if zs_macro_f1 is not None:
        print(f"[Task4] Zero-shot tag prediction macro_f1={zs_macro_f1:.4f} "
              f"(compare against Task 3's supervised macro_f1 in metrics.json)")

    results = {
        "task4_contrastive": {
            "retrieval": retrieval_metrics,
            "zero_shot_tag_macro_f1": zs_macro_f1,
            "history": history,
        }
    }
    dump_metrics(results, os.path.join(cfg["project"]["output_dir"], "metrics.json"))
    return results


@torch.no_grad()
def evaluate_task4(model, loader, device, ks: List[int]):
    model.eval()
    all_audio, all_text = [], []
    for batch in loader:
        audio_emb = model.encode_audio(batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device))
        text_emb = model.encode_text(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_audio.append(audio_emb.cpu().numpy())
        all_text.append(text_emb.cpu().numpy())
    audio_emb = np.concatenate(all_audio)
    text_emb = np.concatenate(all_text)

    sim = audio_emb @ text_emb.T
    a2t = retrieval_recall_at_k(sim, ks=list(ks))
    t2a = retrieval_recall_at_k(sim.T, ks=list(ks))
    return {"audio_to_caption": a2t, "caption_to_audio": t2a}, audio_emb, text_emb


@torch.no_grad()
def zero_shot_tag_eval(model, test_examples, tag_vocab: List[str], device):
    """Encode each test caption + each tag prompt through the (frozen) trained
    text tower, and score tags by cosine similarity — no classifier head.

    test_examples may include DEAM tracks (all-NaN tags_multihot, since DEAM
    has no tag labels — see datasets.build_real_examples). Those rows can't
    contribute to a tag metric no matter how good the model is, and feeding
    NaN ground truth into sklearn crashes rather than silently mishandling it
    — so we filter down to tag-labeled (i.e. FMA-sourced) examples first,
    same as evaluate_task3 already does for the fusion model's tag loss.
    """
    from src.datasets import has_valid_tags

    tag_examples = [ex for ex in test_examples if has_valid_tags(ex)]
    if not tag_examples:
        print("[Task4] zero-shot tag eval skipped: no test examples have tag labels "
              "(e.g. running with a DEAM-only test split)")
        return None

    tag_prompts = [f"a track that is {tag}" for tag in tag_vocab]
    enc_tags = model.text_encoder.tokenize(tag_prompts, device=device)
    tag_emb = model.encode_text(enc_tags["input_ids"], enc_tags["attention_mask"])

    captions = [ex.text for ex in tag_examples]
    enc_cap = model.text_encoder.tokenize(captions, device=device)
    cap_emb = model.encode_text(enc_cap["input_ids"], enc_cap["attention_mask"])

    sims = zero_shot_tag_prediction(cap_emb, tag_emb).cpu().numpy()
    probs = 1 / (1 + np.exp(-10 * (sims - sims.mean())))  # rescale cosine sim into a pseudo-probability
    y_true = np.stack([ex.tags_multihot for ex in tag_examples])
    return compute_tag_metrics(y_true, probs)


def _dump_retrieval_examples(test_examples, audio_emb: np.ndarray, text_emb: np.ndarray, path: str, n: int = 10):
    import json

    sim = audio_emb @ text_emb.T
    examples = []
    n = min(n, len(test_examples))
    for i in range(n):
        top3 = np.argsort(-sim[i])[:3]
        examples.append(
            {
                "query_caption": test_examples[i].text,
                "true_track_id": test_examples[i].track_id,
                "top3_matched_track_ids": [test_examples[j].track_id for j in top3],
                "top3_similarities": [float(sim[i, j]) for j in top3],
            }
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"[Task4] wrote {len(examples)} retrieval examples -> {path}")


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _plot_f1_curve(history: Dict, path: str, title: str, f1_key: str = "val_macro_f1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history["train_loss"], label="train loss", color="tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(history[f1_key], label="val macro-F1", color="tab:blue")
    ax2.set_ylabel("macro-F1", color="tab:blue")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved plot -> {path}")


def _plot_tsne(embeddings, test_examples, path: str):
    if embeddings is None:
        return
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(embeddings), 500)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, n // 3)), random_state=42)
    z2d = tsne.fit_transform(embeddings[:n])
    genres = [test_examples[i].genre_idx for i in range(n)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(z2d[:, 0], z2d[:, 1], c=genres, cmap="tab10", s=25)
    plt.legend(*scatter.legend_elements(), title="genre", loc="best", fontsize=7)
    plt.title("Task 3 fused embeddings (t-SNE, colored by genre)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"saved plot -> {path}")


def _dump_case_studies(model, loader, tag_vocab: List[str], device, path: str, n: int = 3):
    import json

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        out = model(
            batch["graph_x"].to(device), batch["graph_edge_index"].to(device), batch["graph_batch"].to(device),
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
        )
        attn = out["attention"]  # [B, 1, L] or None

    tokenizer = model.text_encoder.tokenizer
    cases = []
    n = min(n, len(batch["track_ids"]))
    for i in range(n):
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
        weights = attn[i, 0].cpu().numpy().tolist() if attn is not None else None
        cases.append({"track_id": batch["track_ids"][i], "tokens": tokens, "attention_weights": weights})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"[Task3] wrote {len(cases)} case studies -> {path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GNN-BERT Music Context Understanding — training entry point")
    parser.add_argument("--task", type=str, required=True, choices=["1", "2", "3", "3-ablations", "4", "all"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="e.g. training.epochs=5 data.mode=real")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    print(f"Using device: {device}")
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)

    t0 = time.time()
    if args.task == "1":
        train_task1(cfg, device)
    elif args.task == "2":
        train_task2(cfg, device)
    elif args.task == "3":
        train_task3(cfg, device, ablation=cfg["task3_fusion"]["fusion_type"])
    elif args.task == "3-ablations":
        run_task3_ablations(cfg, device)
    elif args.task == "4":
        train_task4(cfg, device)
    elif args.task == "all":
        train_task1(cfg, device)
        train_task2(cfg, device)
        run_task3_ablations(cfg, device)
        train_task4(cfg, device)
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()