"""
src/datasets.py

Unified dataset layer for all four tasks. Two data sources are supported,
selected by config['data']['mode']:

  "synthetic" — src/synthetic_data.py generates everything in-memory. No
                downloads needed; use this to develop/debug/demo the pipeline.

  "real"      — loads FMA / MagnaTagATune / MusicCaps / DEAM from data/raw/
                (see Table 1 in the spec for download links). You must
                download these yourself into the paths given in config.yaml;
                this repo does not fetch them automatically.

Every mode produces the same `MusicContextExample` record, so train.py and
evaluate.py are written once against a single interface regardless of source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Batch as PyGBatch
    _TORCH_AVAILABLE = True
except ImportError:  # allows src/datasets.py to be imported for its pure-python
    # helpers (e.g. real-data CSV parsing) in environments without torch.
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

from src.audio_features import SegmentFeatures, segment_track
from src.graph_builder import MusicGraph, build_graph_for_track


@dataclass
class MusicContextExample:
    track_id: str
    graph: MusicGraph
    text: str                      # caption, joined tags, or lyrics — whichever the task needs
    tags_multihot: np.ndarray      # [K]; all-NaN means "no tag labels for this example"
                                    # (e.g. a DEAM track, which has emotion but no tags)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    genre_idx: Optional[int] = None
    source: str = "unknown"        # "fma" | "deam" | "synthetic" — lets callers
                                    # filter to examples that actually have the
                                    # label a given task needs


def has_valid_tags(ex: MusicContextExample) -> bool:
    return not bool(np.isnan(ex.tags_multihot).any())


def has_valid_genre(ex: MusicContextExample) -> bool:
    return ex.genre_idx is not None


def filter_examples(examples: List[MusicContextExample], predicate) -> List[MusicContextExample]:
    return [e for e in examples if predicate(e)]


# -----------------------------------------------------------------------------
# Synthetic source
# -----------------------------------------------------------------------------
def build_synthetic_examples(
    n_tracks: int,
    n_genres: int,
    top_k_tags: int,
    graph_type: str,
    seed: int = 42,
) -> Dict[str, List[MusicContextExample]]:
    """Returns {'train': [...], 'val': [...], 'test': [...]} of
    MusicContextExample, built from src/synthetic_data.py, split artist-
    disjointly."""
    from src.synthetic_data import (
        GENRE_NAMES,
        artist_disjoint_split,
        build_top_k_tag_vocab,
        generate_synthetic_dataset,
        tags_to_multihot,
    )

    tracks = generate_synthetic_dataset(n_tracks=n_tracks, n_genres=n_genres, seed=seed)
    train_t, val_t, test_t = artist_disjoint_split(tracks, seed=seed)
    vocab = build_top_k_tag_vocab(tracks, k=top_k_tags)

    def to_examples(split_tracks):
        examples = []
        for t in split_tracks:
            graph = build_graph_for_track(t.segments, graph_type=graph_type)
            examples.append(
                MusicContextExample(
                    track_id=t.track_id,
                    graph=graph,
                    text=t.caption,
                    tags_multihot=tags_to_multihot(t.tags, vocab),
                    valence=t.valence,
                    arousal=t.arousal,
                    genre_idx=GENRE_NAMES.index(t.genre) if t.genre in GENRE_NAMES else 0,
                )
            )
        return examples

    return {
        "train": to_examples(train_t),
        "val": to_examples(val_t),
        "test": to_examples(test_t),
        "tag_vocab": vocab,
        "genre_vocab": GENRE_NAMES,
    }


# -----------------------------------------------------------------------------
# Real-data source (FMA / MagnaTagATune / MusicCaps / DEAM)
# -----------------------------------------------------------------------------
def load_fma_metadata(metadata_csv: str, keep_subsets=("small", "medium")) -> "pandas.DataFrame":
    """
    FMA ships a multi-index tracks.csv (confirmed header on this project:
    top-level groups album/artist/set/track, e.g. ('track','genre_top'),
    ('track','tags'), ('set','split'), ('set','subset')).

    `keep_subsets` filters to tracks actually present in the archive you
    downloaded — FMA's `subset` column records the SMALLEST archive a track
    belongs to (small subset tracks are also inside fma_medium.zip), so for
    fma_medium.zip keep rows where subset is "small" or "medium".
    """
    import ast
    import pandas as pd

    tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])

    def _get(col_group, col_name, default=""):
        return tracks[(col_group, col_name)] if (col_group, col_name) in tracks.columns else pd.Series(
            default, index=tracks.index
        )

    def _parse_tag_list(raw) -> List[str]:
        """FMA stores tags as a Python-list-literal string, e.g. "['idm', 'electronic']"."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = ast.literal_eval(raw)
            return [str(t) for t in parsed] if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []

    df = pd.DataFrame(
        {
            "track_id": tracks.index.astype(int),
            "genre_top": _get("track", "genre_top"),
            "tags": _get("track", "tags").apply(_parse_tag_list),
            "artist_id": _get("artist", "id", default=tracks.index),
            "split": _get("set", "split", default="training"),
            "subset": _get("set", "subset", default="small"),
        }
    )
    df = df.dropna(subset=["genre_top"])
    df = df[df["genre_top"].astype(str).str.len() > 0]
    if keep_subsets is not None:
        df = df[df["subset"].isin(keep_subsets)]
    return df.reset_index(drop=True)


def load_magnatagatune_annotations(annotations_csv: str, top_k: int = 50) -> "pandas.DataFrame":
    """
    MagnaTagATune ships `annotations_final.csv`: a TSV with one row per clip,
    188 binary tag columns, plus `mp3_path` and `clip_id`. We keep only the
    top-k most frequent tags (standard practice, matches the spec's "top-50
    tags" instruction).
    """
    import pandas as pd

    df = pd.read_csv(annotations_csv, sep="\t")
    tag_cols = [c for c in df.columns if c not in ("clip_id", "mp3_path")]
    top_tags = df[tag_cols].sum(axis=0).sort_values(ascending=False).head(top_k).index.tolist()
    keep_cols = ["clip_id", "mp3_path"] + top_tags
    return df[keep_cols], top_tags


def load_musiccaps_csv(csv_path: str) -> "pandas.DataFrame":
    """MusicCaps ships one CSV with columns including `ytid`, `caption`,
    `aspect_list`, `start_s`, `end_s`. We only need the id + caption here;
    matching audio must be downloaded separately per Google's instructions
    (MusicCaps distributes captions + YouTube IDs, not audio directly)."""
    import pandas as pd

    return pd.read_csv(csv_path)


def load_deam_annotations(annotations_dir: str) -> "pandas.DataFrame":
    """DEAM provides static (song-level) valence/arousal CSVs; we use the
    static annotations (mean over the clip) rather than the 0.5s dynamic
    time series, matching Task 3's per-track regression target.

    DEAM ships these under a nested path:
        <annotations_dir>/annotations/annotations averaged per song/song_level/
    and splits them across two files by song-ID range (e.g.
    static_annotations_averaged_songs_1_2000.csv and ..._2000_2058.csv) since
    the dataset was assembled across multiple annotation years — so we glob
    and concatenate every static_annotations_averaged_songs*.csv found,
    rather than hardcoding one filename.
    """
    import glob
    import pandas as pd

    song_level_dir = os.path.join(annotations_dir, "annotations", "annotations averaged per song", "song_level")
    csv_paths = sorted(glob.glob(os.path.join(song_level_dir, "static_annotations_averaged_songs*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No static_annotations_averaged_songs*.csv found under {song_level_dir} — "
            "check that DEAM_Annotations.zip extracted with its original folder structure intact."
        )

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]  # DEAM's CSVs have known leading-space column names
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined.rename(columns={"song_id": "track_id", "valence_mean": "valence", "arousal_mean": "arousal"})


def build_real_examples(cfg: Dict, graph_type: str = "segment") -> Dict[str, List[MusicContextExample]]:
    """
    Assembles a combined FMA-medium + DEAM example pool.

    Design note (important): FMA and DEAM are DIFFERENT tracks — FMA gives
    tags/genre with no emotion labels, DEAM gives valence/arousal with no
    tags. Rather than requiring every example to have both (impossible here),
    each FMA example gets tags_multihot=<real tags>, valence/arousal=None,
    and each DEAM example gets tags_multihot=<all-NaN>, valence/arousal=<real>.
    fusion_model.fusion_multitask_loss masks NaN rows out of BOTH the tag BCE
    loss and the emotion MSE loss, so a combined batch trains on whichever
    signal each example actually has, and Task 1/Task 2 filter down to
    FMA-only examples via has_valid_tags/has_valid_genre (they have no use
    for a tag-less or genre-less DEAM row).

    Known limitation: DEAM examples get text="" — we didn't require
    downloading DEAM's metadata.zip (which has per-clip genre/tags), so DEAM
    contributes emotion supervision only, not textual signal. Also, DEAM rows
    are split by a deterministic hash of track_id rather than by artist,
    since DEAM's artist info isn't in the annotation files we use — a
    documented deviation from the "no artist leakage" ideal for that portion.
    """
    from src.audio_features import load_audio

    data_cfg = cfg["data"]
    fma_cfg = data_cfg["fma"]
    deam_cfg = data_cfg["deam"]
    top_k_tags = fma_cfg.get("top_k_tags", 50)

    # ---- FMA ----
    metadata = load_fma_metadata(fma_cfg["metadata_csv"], keep_subsets=("small", "medium"))
    genre_vocab = sorted(metadata["genre_top"].astype(str).unique().tolist())
    genre_to_idx = {g: i for i, g in enumerate(genre_vocab)}

    tag_counts: Dict[str, int] = {}
    for tags in metadata["tags"]:
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    tag_vocab = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: -kv[1])[:top_k_tags]]

    fma_examples: List[MusicContextExample] = []
    split_map = {"training": "train", "validation": "val", "test": "test"}
    fma_split_assignment: Dict[str, str] = {}

    for _, row in metadata.iterrows():
        track_id_str = str(int(row["track_id"])).zfill(6)
        audio_path = os.path.join(fma_cfg["audio_dir"], track_id_str[:3], f"{track_id_str}.mp3")
        if not os.path.exists(audio_path):
            continue  # not downloaded, or on FMA's own known-corrupt list

        try:
            y = load_audio(audio_path, sample_rate=fma_cfg["sample_rate"])
            segments = segment_track(y, sample_rate=fma_cfg["sample_rate"], segment_seconds=fma_cfg["segment_seconds"])
            if len(segments) == 0:
                continue
            graph = build_graph_for_track(segments, graph_type=graph_type)

            text = str(row["genre_top"]) + ((": " + ", ".join(row["tags"])) if row["tags"] else "")
            tags_vec = np.zeros(len(tag_vocab), dtype=np.float32)
            for t in row["tags"]:
                if t in tag_vocab:
                    tags_vec[tag_vocab.index(t)] = 1.0

            example = MusicContextExample(
                track_id=str(row["track_id"]),
                graph=graph,
                text=text,
                tags_multihot=tags_vec,
                genre_idx=genre_to_idx.get(str(row["genre_top"])),
                source="fma",
            )
            fma_examples.append(example)
            fma_split_assignment[example.track_id] = split_map.get(str(row["split"]), "train")
        except Exception as e:
            # A handful of FMA files are genuinely corrupt/truncated (a known,
            # documented issue with this dataset) — skip and keep going rather
            # than let one bad file crash a multi-hour run. The whole per-track
            # body is wrapped here, not just audio loading, so a problem in
            # graph-building or tag lookup for one row can't crash the loop
            # either.
            print(f"[build_real_examples] skipping FMA track {row['track_id']} ({audio_path}): {e}")
            continue

    # ---- DEAM ----
    deam_df = load_deam_annotations(deam_cfg["annotations_dir"])
    deam_examples: List[MusicContextExample] = []
    deam_split_assignment: Dict[str, str] = {}

    for _, row in deam_df.iterrows():
        track_id = str(int(row["track_id"]))
        audio_path = os.path.join(deam_cfg["audio_dir"], f"{track_id}.mp3")
        if not os.path.exists(audio_path):
            continue

        try:
            y = load_audio(audio_path, sample_rate=fma_cfg["sample_rate"])
            segments = segment_track(y, sample_rate=fma_cfg["sample_rate"], segment_seconds=fma_cfg["segment_seconds"])
            if len(segments) == 0:
                continue
            graph = build_graph_for_track(segments, graph_type=graph_type)

            example = MusicContextExample(
                track_id=f"deam_{track_id}",
                graph=graph,
                text="",  # see docstring: DEAM's own tags/genre metadata wasn't downloaded
                tags_multihot=np.full(len(tag_vocab), np.nan, dtype=np.float32),
                valence=float(row["valence"]),
                arousal=float(row["arousal"]),
                genre_idx=None,
                source="deam",
            )
            deam_examples.append(example)
            deam_split_assignment[example.track_id] = _hash_split(example.track_id, val_frac=0.15, test_frac=0.15)
        except Exception as e:
            print(f"[build_real_examples] skipping DEAM track {track_id} ({audio_path}): {e}")
            continue

    all_examples = fma_examples + deam_examples
    split_assignment = {**fma_split_assignment, **deam_split_assignment}

    result = {"train": [], "val": [], "test": [], "tag_vocab": tag_vocab, "genre_vocab": genre_vocab}
    for ex in all_examples:
        result[split_assignment.get(ex.track_id, "train")].append(ex)

    print(
        f"[build_real_examples] FMA: {len(fma_examples)} tracks ({len(genre_vocab)} genres, "
        f"{len(tag_vocab)} tags). DEAM: {len(deam_examples)} tracks (emotion only). "
        f"Splits: train={len(result['train'])} val={len(result['val'])} test={len(result['test'])}"
    )
    return result


def _hash_split(track_id: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    """Deterministic pseudo-random split for datasets without an official
    split or usable artist field (see DEAM note in build_real_examples)."""
    import hashlib

    h = int(hashlib.md5(track_id.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < test_frac:
        return "test"
    elif h < test_frac + val_frac:
        return "val"
    return "train"


# -----------------------------------------------------------------------------
# torch Dataset / collate
# -----------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class MusicContextDataset(Dataset):
        def __init__(self, examples: List[MusicContextExample]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> MusicContextExample:
            return self.examples[idx]

    def make_collate_fn(tokenizer, max_length: int = 128):
        """
        Builds a collate_fn that:
          - batches PyG graphs into a single disconnected-union Batch,
          - tokenizes the text field with the given HF tokenizer,
          - stacks tags/valence/arousal into tensors (NaN sentinel for
            missing emotion labels, masked out in fusion_model.fusion_multitask_loss).
        """

        def collate_fn(batch: List[MusicContextExample]):
            pyg_graphs = [ex.graph.to_pyg_data() for ex in batch]
            graph_batch = PyGBatch.from_data_list(pyg_graphs)

            texts = [ex.text for ex in batch]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

            tags = torch.tensor(np.stack([ex.tags_multihot for ex in batch]), dtype=torch.float32)

            valence_arousal = torch.tensor(
                [
                    [ex.valence if ex.valence is not None else float("nan"),
                     ex.arousal if ex.arousal is not None else float("nan")]
                    for ex in batch
                ],
                dtype=torch.float32,
            )

            genre_idx = torch.tensor(
                [ex.genre_idx if ex.genre_idx is not None else -1 for ex in batch], dtype=torch.long
            )

            return {
                "graph_x": graph_batch.x,
                "graph_edge_index": graph_batch.edge_index,
                "graph_batch": graph_batch.batch,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "tags": tags,
                "valence_arousal": valence_arousal,
                "genre_idx": genre_idx,
                "track_ids": [ex.track_id for ex in batch],
            }

        return collate_fn
