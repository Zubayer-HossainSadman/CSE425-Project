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

run the synthetic smoke test end-to-end on
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
