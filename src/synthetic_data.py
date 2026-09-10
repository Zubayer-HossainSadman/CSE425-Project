"""
src/synthetic_data.py

Generates a fully synthetic stand-in for FMA + MagnaTagATune + MusicCaps + DEAM
so the *entire* pipeline (graph construction -> BERT -> GNN -> fusion ->
contrastive) can be built, unit-tested, and demoed end-to-end before the real
datasets are downloaded onto data/raw/ (Table 1 links).

This is not a substitute for running on real data — the deliverables in the
spec (Task 1-4 results tables) require the real datasets — but it lets you:
  1. Verify every module runs without shape/dtype bugs.
  2. Develop and debug training loops quickly on a tiny, fast dataset.
  3. Produce the ">= 20 example .pt/.json graphs" submission artifact
     structurally, even before real audio is available.

Design: each synthetic track secretly belongs to one of `n_genres` archetypes.
Its chroma/MFCC segment features, chord progression, tags, caption, and
valence/arousal are all sampled to be *correlated* with that archetype, so a
model that actually learns something should beat the random-baseline —
letting you sanity-check training curves instead of watching pure noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.audio_features import SegmentFeatures, CHORD_NAMES, build_chord_templates

GENRE_NAMES = [
    "jazz", "rock", "classical", "electronic",
    "hiphop", "folk", "metal", "pop",
]

MOOD_TAGS = [
    "melancholic", "upbeat", "energetic", "calm", "dark", "romantic",
    "nostalgic", "aggressive", "dreamy", "tense", "playful", "epic",
]

INSTRUMENT_TAGS = [
    "piano", "guitar", "synth", "strings", "drums", "vocals",
    "brass", "bass", "sax", "orchestra",
]

ALL_TAGS = GENRE_NAMES + MOOD_TAGS + INSTRUMENT_TAGS  # up to 30; config trims to top_k

CAPTION_TEMPLATES = [
    "A {mood} {genre} track featuring {inst1} and {inst2}.",
    "This {genre} piece has a {mood} atmosphere led by {inst1}.",
    "An {mood} instrumental with prominent {inst1}, {inst2}, and {inst3}.",
    "A {genre} song that feels {mood}, built around {inst1}.",
]


@dataclass
class SyntheticTrack:
    track_id: str
    genre: str
    tags: List[str]
    caption: str
    valence: float
    arousal: float
    segments: List[SegmentFeatures]
    artist_id: str


def _sample_chord_progression(genre_idx: int, n_segments: int, rng: random.Random) -> List[int]:
    """Each genre archetype favors a small pool of chords, so the resulting
    chord-transition graph has genre-correlated structure."""
    templates = build_chord_templates()
    n_chords = templates.shape[0]
    pool_size = 4
    pool_start = (genre_idx * 3) % n_chords
    pool = [(pool_start + i) % n_chords for i in range(pool_size)]
    return [rng.choice(pool) for _ in range(n_segments)]


def _make_segments(chord_ids: List[int], genre_idx: int, track_idx: int, rng: random.Random, noise: float = 0.08) -> List[SegmentFeatures]:
    """
    Bug fix: the previous version derived the MFCC mean from `chord_id % 5`,
    but chord pools are windows of 4 consecutive ids modulo 24 — the mod-5
    vs. mod-24 misalignment meant genre and mfcc-mean were only weakly,
    noisily related. A diagnostic (plain sklearn LogisticRegression on
    mean-pooled MFCC vs. genre) confirmed this: ~0.10 accuracy for 8 classes,
    i.e. no usable signal at all. Fixed by keying the dominant MFCC offset
    directly off `genre_idx` via a fixed per-genre profile, with a smaller
    chord-dependent term layered on top for legitimate within-genre variation.

    A per-track "production/instrumentation" offset is also added: without
    it, averaging many per-segment noise samples over a whole track washes
    the noise out almost completely (noise shrinks by sqrt(n_segments)),
    making genre trivially separable (a first fix attempt hit 1.0 accuracy
    on a plain logistic regression — an unrealistically easy task). The
    per-track term is constant across all of a track's segments, so it
    survives mean-pooling and keeps the task genuinely non-trivial.
    """
    templates = build_chord_templates()
    genre_profile = _GENRE_MFCC_PROFILES[genre_idx % len(_GENRE_MFCC_PROFILES)]
    track_offset = np.random.RandomState(50_000 + track_idx).randn(13) * 1.4
    segments = []
    for i, cid in enumerate(chord_ids):
        chroma = templates[cid] + rng.gauss(0, 1) * noise * np.random.rand(12)
        chroma = np.clip(chroma, 0, None)
        chord_component = np.random.RandomState(cid * 31 + i).randn(13) * 0.6
        mfcc = genre_profile + track_offset + chord_component
        segments.append(SegmentFeatures(start_sec=i * 5.0, end_sec=(i + 1) * 5.0, chroma=chroma, mfcc=mfcc))
    return segments


# Fixed per-genre MFCC "timbre profile" (13-dim), generated once from a
# constant seed so every run of the generator sees the same genre->feature
# mapping. Tuned so genre is learnable-but-not-trivial: a plain logistic
# regression on mean-pooled MFCC should land well above chance (8-class
# random ~0.125) without hitting a suspicious ceiling like 1.0 — see the
# diagnostic in README.md's "What was actually tested" section.
_GENRE_MFCC_PROFILES = np.random.RandomState(999).randn(len(GENRE_NAMES), 13) * 0.9


def generate_synthetic_track(track_idx: int, n_genres: int, rng: random.Random) -> SyntheticTrack:
    genre_idx = track_idx % n_genres
    genre = GENRE_NAMES[genre_idx % len(GENRE_NAMES)]

    n_segments = rng.randint(8, 24)
    chord_ids = _sample_chord_progression(genre_idx, n_segments, rng)
    segments = _make_segments(chord_ids, genre_idx, track_idx, rng)

    n_mood_tags = rng.randint(1, 2)
    n_inst_tags = rng.randint(2, 3)
    moods = rng.sample(MOOD_TAGS, n_mood_tags)
    insts = rng.sample(INSTRUMENT_TAGS, n_inst_tags)
    tags = [genre] + moods + insts

    caption_template = rng.choice(CAPTION_TEMPLATES)
    caption = caption_template.format(
        mood=moods[0], genre=genre,
        inst1=insts[0], inst2=insts[1] if len(insts) > 1 else insts[0],
        inst3=insts[2] if len(insts) > 2 else insts[0],
    )

    # valence/arousal loosely tied to mood word (purely for a plausible synthetic signal)
    high_arousal_moods = {"energetic", "aggressive", "epic", "tense"}
    high_valence_moods = {"upbeat", "playful", "romantic", "dreamy"}
    arousal = 5.0 + (2.5 if moods[0] in high_arousal_moods else -1.5) + rng.gauss(0, 0.7)
    valence = 5.0 + (2.0 if moods[0] in high_valence_moods else -1.0) + rng.gauss(0, 0.7)
    arousal, valence = float(np.clip(arousal, 1, 9)), float(np.clip(valence, 1, 9))

    artist_id = f"artist_{track_idx % max(1, n_genres * 3):03d}"  # a handful of tracks share an artist

    return SyntheticTrack(
        track_id=f"synthetic_{track_idx:05d}",
        genre=genre,
        tags=tags,
        caption=caption,
        valence=valence,
        arousal=arousal,
        segments=segments,
        artist_id=artist_id,
    )


def generate_synthetic_dataset(
    n_tracks: int = 240,
    n_genres: int = 8,
    seed: int = 42,
) -> List[SyntheticTrack]:
    rng = random.Random(seed)
    np.random.seed(seed)
    return [generate_synthetic_track(i, n_genres, rng) for i in range(n_tracks)]


def artist_disjoint_split(
    tracks: List[SyntheticTrack], val_split: float = 0.15, test_split: float = 0.15, seed: int = 42
) -> Tuple[List[SyntheticTrack], List[SyntheticTrack], List[SyntheticTrack]]:
    """
    Splits by artist_id (not by track) so no artist's tracks appear in more
    than one split — implementing the spec's "no artist leakage across
    train/test when possible" requirement (Section 3, step 5).
    """
    rng = random.Random(seed)
    artists = sorted({t.artist_id for t in tracks})
    rng.shuffle(artists)

    n_val_artists = max(1, int(len(artists) * val_split))
    n_test_artists = max(1, int(len(artists) * test_split))

    test_artists = set(artists[:n_test_artists])
    val_artists = set(artists[n_test_artists:n_test_artists + n_val_artists])
    train_artists = set(artists[n_test_artists + n_val_artists:])

    train = [t for t in tracks if t.artist_id in train_artists]
    val = [t for t in tracks if t.artist_id in val_artists]
    test = [t for t in tracks if t.artist_id in test_artists]
    return train, val, test


def build_top_k_tag_vocab(tracks: List[SyntheticTrack], k: int = 20) -> List[str]:
    """Mirrors MagnaTagATune's 'top-50 tags' preprocessing (Task 1 deliverable),
    scaled down to `k` for the synthetic dataset."""
    counts: Dict[str, int] = {}
    for t in tracks:
        for tag in t.tags:
            counts[tag] = counts.get(tag, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:k]
    return [tag for tag, _ in top]


def tags_to_multihot(tags: List[str], vocab: List[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tags:
        if t in vocab:
            vec[vocab.index(t)] = 1.0
    return vec


def synthetic_log_mel(genre_idx: int, track_idx: int, n_mels: int = 128, n_frames: int = 130) -> np.ndarray:
    """
    A fake-but-genre-correlated log-mel spectrogram, used ONLY to exercise the
    CNN baseline (B2, Section 8 / MelSpecCNN in src/gnn_model.py) in synthetic
    mode, since real mel-spectrograms require actual audio. Each genre gets a
    distinct frequency-band emphasis + tempo-like horizontal periodicity so
    the CNN baseline has *something* learnable to compare against the GNN.

    Tuning note (round 2): lowering per-PIXEL noise (round 1) did not fix
    trivial separability — a CNN's pooling/conv layers average over many
    pixels, which cancels iid per-pixel noise just like segment-mean-pooling
    cancelled the earlier MFCC per-segment noise. Diagnosed directly: the
    peak-energy frequency bin formed completely disjoint, non-overlapping
    ranges per genre (e.g. jazz 11-18, rock 23-32, ...) even with the
    round-1 noise level. Fixed the same way as the MFCC bug: added a
    per-TRACK (not per-pixel) jitter to band_center and period, seeded by
    track_idx, so the genre signal survives per-pixel averaging but two
    tracks of adjacent genres can genuinely land in the same frequency band.
    """
    rng = np.random.RandomState(1000 + track_idx)
    freqs = np.linspace(0, 1, n_mels)

    # per-track jitter (not per-pixel): this is what actually creates
    # irreducible ambiguity, since a CNN's pooling would otherwise average
    # away any purely per-pixel noise.
    band_center = (genre_idx + 1) / (len(GENRE_NAMES) + 1) + rng.normal(0, 0.09)
    band_center = float(np.clip(band_center, 0.03, 0.97))
    band = np.exp(-((freqs - band_center) ** 2) / (2 * 0.08 ** 2))  # [n_mels]

    period = max(3.0, 8 + genre_idx * 2 + rng.normal(0, 3.0))
    t = np.arange(n_frames)
    rhythm = 0.5 + 0.5 * np.sin(2 * np.pi * t / period)  # [n_frames]

    mel = np.outer(band, rhythm) * 1.8 - 0.9
    mel += rng.randn(n_mels, n_frames) * 1.3
    mean, std = mel.mean(), mel.std() + 1e-8
    return ((mel - mean) / std).astype(np.float32)
