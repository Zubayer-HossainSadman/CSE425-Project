"""
evaluate.py — assembles the final comparison table (Table 3 in the spec) and a
short markdown summary from whatever results/metrics.json has accumulated
across your train.py runs (Task 1, Task 2 + B1/B2, Task 3 ablations, Task 4).

This does NOT re-run training. Run train.py for each task first:

    python train.py --task 1
    python train.py --task 2
    python train.py --task 3-ablations
    python train.py --task 4
    python evaluate.py

Each train.py call appends to results/metrics.json (it merges rather than
overwrites — see src.utils.dump_metrics), so run them in any order/subset and
evaluate.py will summarize whatever is present, flagging missing entries
rather than failing.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional


def _get(d: Dict, *path, default="—"):
    node = d
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def build_comparison_table(metrics: Dict) -> str:
    """Recreates the shape of Table 3 ('Illustrative performance comparison')
    from actual run results, rather than the spec's placeholder numbers."""
    rows = []

    b1 = metrics.get("baseline_b1_majority", {})
    rows.append(("Random / majority tags (B1)", _get(b1, "test_macro_f1"), "—", "—", "—"))

    b2 = metrics.get("baseline_b2_cnn_melspec", {})
    rows.append(("CNN mel-spec (B2)", _get(b2, "test_macro_f1"), "—", "—", "—"))

    t1 = metrics.get("task1_bert", {})
    rows.append(("Task 1: BERT-only", _get(t1, "test_macro_f1"), _get(t1, "test_auc_pr"), "—", "—"))

    t2 = metrics.get("task2_gnn", {})
    rows.append(("Task 2: GNN-only", _get(t2, "test_macro_f1"), "—", "—", "—"))

    for key, label in [
        ("task3_bert_only", "Task 3 ablation: BERT-only"),
        ("task3_gnn_only", "Task 3 ablation: GNN-only"),
        ("task3_concat", "Task 3 ablation: early concat"),
        ("task3_cross_attention", "Task 3: GNN-BERT (cross-attn)"),
    ]:
        t3 = metrics.get(key, {})
        rows.append((label, _get(t3, "tag_macro_f1"), _get(t3, "tag_auc_pr"), _get(t3, "emotion_mae"), "—"))

    t4 = metrics.get("task4_contrastive", {})
    r_at_5 = _get(t4, "retrieval", "caption_to_audio", "R@5")
    rows.append(("Task 4: Contrastive retrieval", _get(t4, "zero_shot_tag_macro_f1"), "—", "—", r_at_5))

    header = f"| {'Model':32s} | {'Macro-F1':9s} | {'AUC-PR':7s} | {'MAE (emotion)':13s} | {'R@5 (retrieval)':15s} |"
    sep = "|" + "-" * 34 + "|" + "-" * 11 + "|" + "-" * 9 + "|" + "-" * 15 + "|" + "-" * 17 + "|"
    lines = [header, sep]
    for name, f1, auc, mae, r5 in rows:
        f1_s = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        mae_s = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
        r5_s = f"{r5:.4f}" if isinstance(r5, float) else str(r5)
        lines.append(f"| {name:32s} | {f1_s:9s} | {auc_s:7s} | {mae_s:13s} | {r5_s:15s} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Assemble the final comparison table from results/metrics.json")
    parser.add_argument("--metrics_path", type=str, default="results/metrics.json")
    parser.add_argument("--output_path", type=str, default="results/comparison_table.md")
    args = parser.parse_args()

    if not os.path.exists(args.metrics_path):
        print(f"No metrics found at {args.metrics_path} yet — run train.py for at least one task first.")
        return

    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)

    table_md = build_comparison_table(metrics)
    print("\n" + table_md + "\n")
    print("(entries showing '—' mean that task/baseline hasn't been run yet — see the docstring at the top of this file)")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write("# Model comparison (auto-generated from results/metrics.json)\n\n")
        f.write(table_md + "\n")
    print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
