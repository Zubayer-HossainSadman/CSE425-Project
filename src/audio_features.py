"""
src/audio_features.py

Section 3 "Preprocessing pipeline" step 1-2:
  1. Resample to 22,050 Hz; extract log-mel spectrogram (128 bins) or chroma
     (12 bins); normalize per track.
  2. Split each track into fixed windows (5-10s) or beat-synchronous segments.

This module wraps librosa. It is imported lazily (inside functions) so the
rest of the codebase can be imported/tested even where librosa/audio codecs
are not installed (e.g. this sandbox) — only code paths that actually touch
raw audio files need librosa at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SegmentFeatures:
    """Per-segment features for one track — the node features that feed
    graph_builder.py."""

    start_sec: float
    end_sec: float
    chroma: np.ndarray      # [12] mean chroma vector for this segment
    mfcc: np.ndarray        # [13] mean MFCC vector for this segment (for similarity edges)
    mel: Optional[np.ndarray] = None  # [128, T_seg] optional full log-mel patch


def load_audio(path: str, sample_rate: int = 22050) -> np.ndarray:
    """Load and resample audio to the target sample rate (mono).

    Also validates the decode: some FMA/DEAM files are corrupted in ways
    that don't raise a clean exception (the mpg123 backend just spams
    "dequantization failed" warnings and returns a short/garbage waveform).
    Raising here, immediately after decode, gives build_real_examples' `except
    Exception` a clear, early, single point to catch and skip the track —
    instead of a corrupted `y` silently propagating into chroma/MFCC
    extraction and surfacing as a confusing failure deep inside numpy several
    function calls later.
    """
    import librosa

    y, _sr = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("decoded to zero samples (empty/corrupted audio file)")
    if not np.isfinite(y).all():
        raise ValueError("decoded audio contains NaN/Inf samples (corrupted audio file)")
    if y.size < sample_rate:  # under 1 second of audio isn't usable for segmentation
        raise ValueError(f"decoded audio too short ({y.size} samples, ~{y.size / sample_rate:.2f}s)")
    return y


def extract_log_mel(
    y: np.ndarray, sample_rate: int = 22050, n_mels: int = 128, hop_length: int = 512
) -> np.ndarray:
    """Log-mel spectrogram, per-track normalized to zero mean / unit variance."""
    import librosa

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return _normalize(log_mel)


def extract_chroma(
    y: np.ndarray, sample_rate: int = 22050, n_chroma: int = 12, hop_length: int = 512
) -> np.ndarray:
    """Chroma-STFT features, per-track normalized. Shape [n_chroma, T]."""
    import librosa

    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate, n_chroma=n_chroma, hop_length=hop_length)
    return _normalize(chroma)


def extract_mfcc(y: np.ndarray, sample_rate: int = 22050, n_mfcc: int = 13, hop_length: int = 512) -> np.ndarray:
    """MFCCs, used for segment-similarity edges in graph_builder.py."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    return _normalize(mfcc)


def _normalize(feat: np.ndarray) -> np.ndarray:
    mean, std = feat.mean(), feat.std() + 1e-8
    return (feat - mean) / std


def segment_track(
    y: np.ndarray,
    sample_rate: int = 22050,
    segment_seconds: float = 5.0,
    hop_length: int = 512,
    beat_synchronous: bool = False,
) -> List[SegmentFeatures]:
    """
    Step 2 of the preprocessing pipeline: split a track into fixed 5-10s
    windows, OR beat-synchronous segments (librosa.beat.beat_track), and
    compute per-segment chroma/MFCC features (the node features used to
    build both the chord-transition graph and the segment-similarity graph
    in graph_builder.py).
    """
    import librosa

    chroma_full = extract_chroma(y, sample_rate, hop_length=hop_length)
    mfcc_full = extract_mfcc(y, sample_rate, hop_length=hop_length)

    # chroma_stft and mfcc can produce slightly different frame counts on
    # edge-case audio (this is what actually crashed on a corrupted FMA file:
    # a mask built from chroma_full's frame count was applied to mfcc_full,
    # which had a different number of columns). Truncate both to the shorter
    # length so the mask below is always valid for both arrays.
    min_frames = min(chroma_full.shape[1], mfcc_full.shape[1])
    chroma_full = chroma_full[:, :min_frames]
    mfcc_full = mfcc_full[:, :min_frames]
    n_frames = min_frames

    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sample_rate, hop_length=hop_length)

    if beat_synchronous:
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, hop_length=hop_length)
        boundaries = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
        boundaries = np.concatenate([[0.0], boundaries, [frame_times[-1] if n_frames else 0.0]])
    else:
        total_seconds = frame_times[-1] if n_frames else 0.0
        n_segments = max(1, int(np.ceil(total_seconds / segment_seconds)))
        boundaries = np.arange(n_segments + 1) * segment_seconds

    segments: List[SegmentFeatures] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        # Last line of defense: even with the frame-count fix above and the
        # load_audio validation, a pathological file could still produce a
        # near-empty masked slice. Suppress numpy's warning-as-side-effect
        # here rather than let a RuntimeWarning about an empty/NaN slice risk
        # being escalated to an error by whatever warnings filter is active.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            mask = (frame_times >= start) & (frame_times < end)
            if not mask.any():
                continue
            chroma_seg = chroma_full[:, mask].mean(axis=1)
            mfcc_seg = mfcc_full[:, mask].mean(axis=1)
            if not (np.isfinite(chroma_seg).all() and np.isfinite(mfcc_seg).all()):
                continue  # skip a degenerate segment rather than poison the graph with NaN
            segments.append(
                SegmentFeatures(start_sec=float(start), end_sec=float(end), chroma=chroma_seg, mfcc=mfcc_seg)
            )
    return segments


def dominant_chord_labels(segments: List[SegmentFeatures], chord_templates: Optional[np.ndarray] = None) -> List[int]:
    """
    Cheap chord-labeling by template matching against the 12 major + 12 minor
    triad templates in chroma space (a lightweight stand-in for a full chord
    recognizer, sufficient for building the chord-transition graph in
    Section 3's "Chord-transition graph" bullet).

    Returns one chord-class index (0-23: 12 major, 12 minor) per segment.
    """
    if chord_templates is None:
        chord_templates = build_chord_templates()
    labels = []
    for seg in segments:
        v = seg.chroma / (np.linalg.norm(seg.chroma) + 1e-8)
        scores = chord_templates @ v
        labels.append(int(np.argmax(scores)))
    return labels


def build_chord_templates() -> np.ndarray:
    """24 binary chroma templates: 12 major triads + 12 minor triads,
    each rotated from a root-position [1,0,0,1,0,0,0,1,0,0,0,0] (major)
    or [1,0,0,1,0,0,0,0,1,0,0,0]-style (minor) pitch-class pattern."""
    major_intervals = [0, 4, 7]
    minor_intervals = [0, 3, 7]
    templates = []
    for root in range(12):
        maj = np.zeros(12)
        for i in major_intervals:
            maj[(root + i) % 12] = 1.0
        templates.append(maj)
    for root in range(12):
        minr = np.zeros(12)
        for i in minor_intervals:
            minr[(root + i) % 12] = 1.0
        templates.append(minr)
    templates = np.stack(templates, axis=0)
    templates = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-8)
    return templates


CHORD_NAMES = [f"{n}maj" for n in "C C# D D# E F F# G G# A A# B".split()] + [
    f"{n}min" for n in "C C# D D# E F F# G G# A A# B".split()
]
