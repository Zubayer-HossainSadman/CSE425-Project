# Model comparison (auto-generated from results/metrics.json)

| Model                            | Macro-F1  | AUC-PR  | MAE (emotion) | R@5 (retrieval) |
|----------------------------------|-----------|---------|---------------|-----------------|
| Random / majority tags (B1)      | 0.0270    | —       | —             | —               |
| CNN mel-spec (B2)                | 0.2853    | —       | —             | —               |
| Task 1: BERT-only                | 0.7814    | 0.9587  | —             | —               |
| Task 2: GNN-only                 | 0.2232    | —       | —             | —               |
| Task 3 ablation: BERT-only       | 0.6279    | 0.8471  | —             | —               |
| Task 3 ablation: GNN-only        | 0.0315    | 0.3218  | —             | —               |
| Task 3 ablation: early concat    | 0.4031    | 0.7090  | 0.6869        | —               |
| Task 3: GNN-BERT (cross-attn)    | 0.0000    | 0.3950  | 0.6268        | —               |
| Task 4: Contrastive retrieval    | 0.0287    | —       | —             | 0.0112          |
