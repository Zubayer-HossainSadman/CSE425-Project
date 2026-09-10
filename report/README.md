# Final report

Place `final_report.pdf` (6–10 pages) here, written from one of the templates
linked in the top-level `README.md`.

Suggested section mapping to the spec's grading rubric (Section 9, Table 4):

1. **Introduction & Motivation** — condense Section 1 of the spec in your own words.
2. **Problem Definition** — restate the formalization from Section 2 (T = (X_audio, X_text, G, y)).
3. **Datasets & Preprocessing** (15%) — which of Table 1's datasets you used, your
   train/val/test split, and confirmation of no artist leakage (cite
   `src/synthetic_data.artist_disjoint_split` or FMA's `set.split` column).
4. **Model Implementation** (25%) — one subsection per task (1–4), referencing
   the equations in Section 4 and the corresponding `src/*.py` file.
5. **Results** (20% context quality + 15% baselines + 15% metrics/analysis) —
   the table `results/comparison_table.md` (from `evaluate.py`), F1 curves
   (`results/plots/`), the t-SNE plot, 3 case studies, and 10 retrieval examples.
6. **Discussion** — what worked, what didn't, and why (e.g. does cross-attention
   actually beat early-concat on your data? does the GNN alone struggle where
   BERT alone doesn't, or vice versa?).
7. **Conclusion & Future Work.**

Do not include synthetic-data numbers in the final report as experimental
results — they exist only to verify the pipeline runs; re-run everything on
real data (see the top-level README's "Switching to real data" section)
before writing this document.
