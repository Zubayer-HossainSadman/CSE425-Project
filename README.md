# GNN-Based BERT for Understanding Context from Music

Implementation scaffold for the CSE425/EEE474/CSE715 Neural Networks project
*"GNN-Based BERT for Understanding Context from Music"* (deadline: 2 Oct 2026).

Builds a hybrid **BERT + GNN** system across the four required tasks:

| Task | What it does | Key file |
|---|---|---|
| 1 (Easy) | BERT multi-label tag/caption classifier | `src/bert_encoder.py` |
| 2 (Medium) | GraphSAGE/GAT on chord/segment graphs -> genre | `src/gnn_model.py` |
| 3 (Hard) | Cross-attention GNN-BERT fusion -> tags + emotion | `src/fusion_model.py` |
| 4 (Advanced) | Contrastive GNN-BERT dual encoder -> retrieval | `src/contrastive.py` |

## ⚠️ Update: two real bugs found and fixed after the first GPU run

The first end-to-end run on real hardware (thank you for that — it's exactly
what this sandbox couldn't do) surfaced two genuine bugs, confirmed with
targeted diagnostics and now fixed in this version:

1. **Task 3 fusion loss was scale-broken.** Valence/arousal targets are raw
   1-9 scale; squared error against an untrained regression head is ~30,
   which at `alpha=beta=0.5` completely swamped the ~0.6 tag-BCE loss (observed:
   initial loss ~19.5 for `concat`/`cross_attention` vs. ~0.53 for `bert_only`).
   The tag head never got a usable gradient before early stopping. **Fix:**
   `src/utils.EmotionStats` z-scores valence/arousal from the train split
   before they enter the loss, and denormalizes predictions before computing
   MAE/R² — confirmed by hand-calculation to bring the initial loss down to
   ~1.6, in line with `bert_only`.
2. **Task 2's synthetic MFCC features had no real genre signal.** A
   diagnostic (plain sklearn `LogisticRegression` on mean-pooled MFCC vs.
   genre) scored ~0.10 accuracy for 8 classes — random chance — explaining
   why the GNN never beat the majority baseline and the CNN mel-spec baseline
   (built with a much more obviously genre-correlated signal) looked
   artificially strong by comparison, even hitting a suspicious 1.0 macro-F1.
   **Fix:** `src/synthetic_data._make_segments` now keys the MFCC mean
   directly off `genre_idx` via a fixed per-genre profile (previously it used
   `chord_id % 5`, which didn't align with the chord pools' `mod 24`
   structure), plus a per-track offset so genre survives segment-mean-pooling
   without being trivially separable. Re-diagnosed at accuracy 0.67 / macro-F1
   0.29 — learnable, not random, not suspiciously perfect. The CNN baseline's
   signal-to-noise ratio was also toned down to match (it previously hit a
   suspicious macro-F1 = 1.0000).

**If you already ran `--task 2` or `--task 3`/`3-ablations` on the old
version, re-run them** — the old numbers reflected these bugs, not the real
models.

## ⚠️ Update 2: two more issues found on the second GPU run

1. **B2 (CNN mel-spec) was still trivially separable after the round-1 fix.**
   Lowering per-*pixel* noise didn't help — a CNN's conv/pooling layers
   average over many pixels, which cancels iid per-pixel noise the same way
   segment-mean-pooling cancelled the earlier MFCC noise. Diagnosed directly:
   each genre's peak-energy frequency bin formed a completely disjoint range
   even after round 1 (e.g. jazz 11-18, rock 23-32, ...). **Fix:**
   `synthetic_log_mel` now adds a per-*track* jitter to `band_center`/`period`
   (not per-pixel), so genre bands genuinely overlap between tracks —
   re-diagnosed with overlapping ranges across all 8 genres.
2. **`concat`/`cross_attention` still scored macro-F1 = 0.0000 even after the
   loss-scale fix**, despite the initial loss now landing exactly where
   predicted (~1.68, matching the ~1.6 hand-calculation). Root cause: early
   stopping and best-checkpoint selection were keyed on macro-F1, which
   thresholds probabilities at 0.5 and reads a hard 0.0000 for every epoch
   until the model's calibration crosses that threshold — Task 1's BERT
   needed 4-5 epochs to cross it. With `early_stopping_patience: 5`, the
   fusion models ran out of patience budget before ever calibrating. **Fix:**
   `train_task1` and `train_task3` now select the best checkpoint (and count
   patience) by AUC-PR, which is threshold-free and moves from epoch 1. Task
   2 is unaffected — argmax classification doesn't have this stall mode.

## ⚠️ Update 3: real-data loading (FMA-medium + DEAM) is now actually implemented

The previous version's `build_real_examples` was a rough placeholder — it
never built genre labels (Task 2 would have silently had nothing to train
on), never touched DEAM, and used a `<track_id>.mp3` audio path that doesn't
match FMA's real on-disk layout. Now implemented and verified against the
project's actual downloaded file headers/structure:

- **FMA path fix:** FMA nests audio by the first 3 digits of the zero-padded
  6-digit track ID (`fma_medium/000/000002.mp3`), not flat `<id>.mp3`.
- **FMA metadata parsing:** `load_fma_metadata` now reads the real multi-index
  `tracks.csv` header (`album.*`/`artist.*`/`set.split`/`set.subset`/`track.*`),
  parses the `track.tags` Python-list-literal string, builds a genre
  vocabulary from `genre_top`, and filters to `subset in {small, medium}` so
  only tracks actually present in `fma_medium.zip` are kept.
- **DEAM loading (new):** `load_deam_annotations` globs and concatenates
  DEAM's two `static_annotations_averaged_songs_*.csv` range files (the
  dataset splits them by song-ID range across annotation years) from
  `annotations/annotations averaged per song/song_level/`.
- **Combined FMA+DEAM training (new, important):** FMA and DEAM are
  *different tracks* — FMA has tags/genre with no emotion labels, DEAM has
  emotion labels with no tags. Each `MusicContextExample` now carries a NaN
  sentinel for whichever label its source dataset doesn't provide, and
  `fusion_multitask_loss` (plus the `bert_only`/`gnn_only` ablations' direct
  BCE calls, and `evaluate_task3`'s metric computation) mask those NaN rows
  out. **This masking is required correctness, not a refinement** — PyTorch's
  default mean reduction propagates a single NaN to poison an entire batch's
  loss the moment FMA and DEAM rows land in the same batch; without it,
  Task 3 would silently NaN out the first time real data was used.
- **Task 1 / Task 2 filtering (new):** these tasks now request
  `get_splits(..., require="tags")` / `require="genre"` respectively, which
  drops DEAM's tag-less/genre-less examples before training — they only
  learn from FMA.

**Known, documented limitations of this exact pairing** (not bugs — inherent
to choosing FMA+DEAM specifically, without MagnaTagATune/MusicCaps):
- DEAM examples get `text=""` — DEAM's own per-clip genre/tags live in
  `metadata.zip`, which wasn't required for this setup, so DEAM contributes
  emotion supervision only, never textual signal.
- DEAM rows are split by a deterministic hash of `track_id`
  (`datasets._hash_split`), not by artist — DEAM's annotation files don't
  include artist info, so true artist-disjoint splitting (Section 3, step 5)
  only holds for the FMA portion of the data, which uses FMA's own official
  `set.split` column (already artist-disjoint by construction).
- Task 4 (contrastive retrieval) is not meaningfully usable with this pairing
  — it needs natural-language captions, which neither FMA's tags nor DEAM
  provide. The spec marks Task 4 as "optional → bonus," so this is a
  legitimate, complete choice for the core 100-point rubric.

None of this has been run against the actual downloaded files yet — only
against mock CSVs built to match the *exact* headers confirmed from this
project's real `tracks.csv` and DEAM annotation files. Run `--task 1` on a
small scale first (see below) before committing to a full run.

## ⚠️ Update 4: crash on a corrupted FMA file, fixed at the source

A real run on `--task 1 --override data.mode=real` hit an unhandled crash
inside `segment_track`'s `.mean(axis=1)` call, triggered by one specific FMA
file that produced dozens of `mpg123 ... dequantization failed!` decoder
warnings before decoding to something degenerate. Two real bugs, both fixed:

1. **`chroma_stft` and `mfcc` can return different frame counts** on edge-case
   audio. `segment_track` built its segment mask from `chroma_full`'s frame
   count but applied that same mask to `mfcc_full` — on a corrupted file
   where the two extractors disagreed on frame count, this was the actual
   crash site. **Fix:** both arrays are now truncated to their shared minimum
   length before any masking happens, so this can't occur regardless of why
   the counts differed.
2. **No validation after decode.** A corrupted mp3 can decode to something
   technically non-empty but degenerate (NaN/Inf samples, or near-silence) —
   `load_audio` now checks for exactly this (empty, non-finite, or under 1
   second) and raises a clear, specific error immediately after decoding, so
   `build_real_examples`'s per-track `except Exception` catches and skips it
   with an informative message instead of the failure surfacing three
   function calls later inside numpy with no context. `segment_track` also
   now drops any individual segment whose computed mean isn't finite, as a
   second layer of defense.

This is the second time real hardware surfaced something a sandbox-only pass
couldn't have found (corrupted third-party audio files aren't something you
can synthesize a test case for in advance) — worth remembering if a future
run hits yet another edge case in some other corrupted file this dataset
ships with.

## Important: run this on your GPU machine, not in a sandbox

This repo was authored and syntax-checked in an environment **without internet
access and without `torch`/`transformers`/`torch-geometric`/`librosa`
installed**, so the deep-learning code paths could not be executed end-to-end
here. Every module was:
- written against the exact, standard APIs of PyTorch / HuggingFace
  Transformers / PyTorch Geometric / librosa,
- unit-tested wherever it only needed numpy/pandas/sklearn/networkx (graph
  construction, chord-template matching, artist-disjoint splitting, all
  metrics, config loading, the synthetic data generator — see "What was
  actually tested" below),
- but **not** run as a full training loop.

**Before you trust any numbers**, run the synthetic smoke test end-to-end on
your machine first (RTX 4090 / 24GB VRAM is plenty):

```bash
pip install -r requirements.txt
python train.py --task 1 --override training.epochs=2   # ~1 min sanity check
python train.py --task 2 --override training.epochs=2
python train.py --task 3 --override training.epochs=2
python train.py --task 4 --override training.epochs=2
python evaluate.py
```

If something breaks, it's most likely a minor shape/API mismatch from a
library-version difference — the architecture and math (matching the spec's
equations 1:1) are the parts you can trust; treat the exact tensor plumbing as
"needs your first real run to shake out."

## What was actually tested in this sandbox

Ran and verified correct output for:
- `src/utils.py` — all metrics (Macro/Micro-F1, AUC-PR, MAE/R², retrieval
  R@K, graph coherence score), config loading + CLI overrides.
- `src/audio_features.py` — chord-template construction & matching (verified
  a synthetic C-major segment is correctly classified as `Cmaj`).
- `src/graph_builder.py` — chord-transition graph correctly links C↔G given
  an alternating progression; segment-similarity graph correctly finds the
  two similarity clusters in a toy example.
- `src/synthetic_data.py` — dataset generation, artist-disjoint splitting
  (confirmed zero artist overlap across train/val/test), tag vocab building.
- `evaluate.py` — comparison-table generation from a mock `metrics.json`.
- Every `.py` file — syntax-checked with `py_compile` (no import errors for
  anything not requiring the missing GPU-stack packages).

**Not executable here** (needs `torch`/`transformers`/`torch-geometric`/
`librosa`, none of which could be installed without network access):
model forward/backward passes, the training loops in `train.py`, real audio
loading in `src/audio_features.py`.

## Quickstart (synthetic data — no downloads needed)

```bash
pip install -r requirements.txt

# smoke-test every task on generated data (config.yaml: data.mode=synthetic)
python train.py --task 1
python train.py --task 2
python train.py --task 3-ablations     # bert-only / gnn-only / concat / cross-attention
python train.py --task 4
python evaluate.py                     # builds results/comparison_table.md

# generate the >=20 example preprocessed graphs required at submission
python -m src.export_example_graphs --n_tracks 12
```

Or open `notebooks/demo_context.ipynb` for the one-end-to-end-inference-call
demo (graph + caption in -> tags + emotion out) required by Section 10.

## Switching to real data

1. Download into `data/raw/` per Table 1 of the spec:
   [FMA](https://github.com/mdeff/fma), [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset),
   [GTZAN](http://marsyas.info/downloads/datasets.html), [DEAM](https://cvml.unige.ch/databases/DEAM/),
   [MusicCaps](https://huggingface.co/datasets/google/MusicCaps), [MSD](http://millionsongdataset.com/),
   [Lakh MIDI](https://colinraffel.com/projects/lmd/), [EmoMusic](https://cvml.unige.ch/databases/emoMusic/).
2. Edit the paths under `data:` in `config.yaml` to match where you put them.
3. Set `data.mode: real`.
4. `src/datasets.py`'s `load_fma_metadata` / `load_magnatagatune_annotations` /
   `load_musiccaps_csv` / `load_deam_annotations` and `build_real_examples`
   document the *expected* on-disk layout and column names — adjust them to
   whatever exact dump you downloaded (dataset packaging changes over time,
   e.g. FMA's `tracks.csv` header format).
5. No artist leakage: FMA's own `set.split` column is already artist-disjoint;
   `src/synthetic_data.artist_disjoint_split` shows the same logic for any
   dataset that isn't pre-split this way.

## Project structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml                # single source of truth for every hyperparameter/path
├── data/
│   ├── raw/                   # put downloaded FMA/MagnaTagATune/MusicCaps/DEAM here
│   ├── processed/             # graphs, mel-spec/BERT caches (export_example_graphs.py output lives here)
│   └── splits/                # train/val/test JSON, if you materialize them
├── notebooks/
│   ├── eda.ipynb              # genre/tag distributions, chroma heatmap
│   └── demo_context.ipynb     # required end-to-end inference demo
├── src/
│   ├── audio_features.py      # mel/chroma/MFCC extraction, segmentation, chord templates
│   ├── graph_builder.py       # chord-transition + segment-similarity graph construction
│   ├── bert_encoder.py        # BERT wrapper + Task 1 classifier
│   ├── gnn_model.py           # GraphSAGE/GAT encoder + Task 2 classifier + CNN baseline
│   ├── fusion_model.py        # Task 3 cross-attention / concat fusion
│   ├── contrastive.py         # Task 4 InfoNCE dual encoder
│   ├── datasets.py            # unified real/synthetic dataset + PyG collate_fn
│   ├── synthetic_data.py      # dependency-free synthetic dataset generator
│   ├── export_example_graphs.py
│   └── utils.py                # seeding, config, checkpointing, ALL Section-6 metrics
├── train.py                   # CLI: --task {1,2,3,3-ablations,4,all}
├── evaluate.py                # builds the Table-3-style comparison from results/metrics.json
├── results/
│   ├── metrics.json           # accumulated across train.py runs
│   ├── plots/                 # F1 curves, t-SNE
│   └── retrieval_examples/    # Task 4 qualitative examples
└── report/                    # write final_report.pdf here (see template links below)
```

## Report template links

- NeurIPS 2024: https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh
- IEEE Conference: https://www.overleaf.com/latex/templates/ieee-conference-template
- ICML 2025: https://www.overleaf.com/latex/templates/icml2025-template

## Grading-rubric checklist (Section 9-10 of the spec)

- [x] Dataset & preprocessing code, artist-disjoint splitting logic
- [x] BERT (Task 1), GraphSAGE/GAT (Task 2), fusion (Task 3), contrastive (Task 4) implementations
- [x] Two required baselines: B1 (majority-class) and B2 (CNN on mel-spectrogram)
- [x] Metrics: Macro/Micro-F1, AUC-PR, MAE/R², retrieval R@K, graph coherence score
- [x] Ablations: BERT-only / GNN-only / early-concat / cross-attention (`--task 3-ablations`)
- [x] >=20 example preprocessed graphs (`src/export_example_graphs.py`)
- [x] Demo notebook with one end-to-end inference example
- [ ] **Run everything on real FMA/MagnaTagATune/MusicCaps/DEAM data** — synthetic
      results are for pipeline verification only and are **not** valid experimental
      results for the report.
- [ ] t-SNE plot, 3 case studies, 10 qualitative retrieval examples — code is wired
      up in `train.py` and writes these automatically during a real training run.
- [ ] Final report PDF (6-10 pages, NeurIPS/IEEE/ICML template) — write this yourself
      once you have real numbers; this repo does not fabricate a report.
