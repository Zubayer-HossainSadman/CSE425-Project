"""
src/export_example_graphs.py

Submission requirement #2: "Preprocessed graph samples (at least 20 example
.pt / .json graphs)". This script builds both a chord-transition graph and a
segment-similarity graph for a handful of tracks (synthetic by default; point
it at real audio via --mode real once FMA/etc. are downloaded) and writes
them to data/processed/ as human-readable .json (always) and, if torch is
installed, as .pt PyG Data objects too.

Usage:
    python -m src.export_example_graphs --n_tracks 12 --mode synthetic
    # -> 12 tracks x 2 graph types = 24 graph samples, comfortably over the
    #    "at least 20" requirement.
"""

from __future__ import annotations

import argparse
import json
import os

from src.graph_builder import build_graph_for_track


def export_synthetic_graphs(n_tracks: int, out_dir: str) -> int:
    from src.synthetic_data import generate_synthetic_dataset

    os.makedirs(out_dir, exist_ok=True)
    tracks = generate_synthetic_dataset(n_tracks=n_tracks, n_genres=8, seed=123)

    count = 0
    for track in tracks:
        for graph_type in ("chord", "segment"):
            graph = build_graph_for_track(track.segments, graph_type=graph_type)

            json_path = os.path.join(out_dir, f"{track.track_id}_{graph_type}.json")
            with open(json_path, "w") as f:
                json.dump(graph.to_json_dict(), f, indent=2)
            count += 1

            try:
                import torch

                pt_path = os.path.join(out_dir, f"{track.track_id}_{graph_type}.pt")
                torch.save(graph.to_pyg_data(), pt_path)
            except ImportError:
                pass  # .json export still satisfies the submission requirement

    return count


def main():
    parser = argparse.ArgumentParser(description="Export example preprocessed graphs to data/processed/")
    parser.add_argument("--n_tracks", type=int, default=12, help="tracks x 2 graph types must be >= 20")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    if args.mode == "real":
        raise NotImplementedError(
            "Wire this up to src.datasets.build_real_examples once FMA/etc. are on disk; "
            "iterate its output list and call graph.to_json_dict()/to_pyg_data() the same way."
        )

    n = export_synthetic_graphs(args.n_tracks, args.out_dir)
    print(f"Exported {n} graph samples to {args.out_dir}/")


if __name__ == "__main__":
    main()
