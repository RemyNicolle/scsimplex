# scsimplex benchmark summary

## Sample metrics
| sample              | n_cells  | n_genes   | n_metacells | compression_ratio | kmeans_time_sec | mean_loglik_kmeans | mean_loglik_cellstates | ari_vs_cellstates | nmi_vs_cellstates | bootstrap_cophenetic_log1p_mean | bootstrap_cophenetic_log1p_sd | bootstrap_cophenetic_clr_mean | bootstrap_cophenetic_clr_sd | pseudobulk_rows |
| ------------------- | -------- | --------- | ----------- | ----------------- | --------------- | ------------------ | ---------------------- | ----------------- | ----------------- | ------------------------------- | ----------------------------- | ----------------------------- | --------------------------- | --------------- |
| RNAmatrix_Hajk_S01  | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0146          | -5958.7888         | -5711.0811             | 0.0760            | 0.5121            | 0.8312                          | 0.0219                        | 0.7980                        | 0.0276                      | 60.0000         |
| RNAmatrix_Hajk_S02  | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0089          | -3886.8926         | -3556.5361             | -0.0002           | 0.3286            | 0.7584                          | 0.0818                        | 0.7671                        | 0.0245                      | 60.0000         |
| RNAmatrix_Hajk_S03  | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0137          | -6355.6542         | -6107.9846             | 0.0267            | 0.4499            | 0.7486                          | 0.0225                        | 0.7603                        | 0.0257                      | 60.0000         |
| RNAmatrix_Hajk_S04  | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0084          | -5678.3376         | -5135.5348             | -0.0002           | 0.4498            | 0.7973                          | 0.0484                        | 0.6533                        | 0.1057                      | 60.0000         |
| RNAmatrix_Hajk_S05  | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0195          | -6806.2078         | NaN                    | NaN               | NaN               | 0.6761                          | 0.0636                        | 0.6987                        | 0.0659                      | 60.0000         |
| RNAmatrix_HtanZ_S59 | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0120          | -3228.8678         | NaN                    | NaN               | NaN               | 0.6182                          | 0.0232                        | 0.5253                        | 0.0156                      | 60.0000         |
| RNAmatrix_HtanZ_S60 | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0071          | -2671.6149         | NaN                    | NaN               | NaN               | 0.7720                          | 0.0405                        | 0.7138                        | 0.0225                      | 60.0000         |
| RNAmatrix_HtanZ_S61 | 300.0000 | 3000.0000 | 60.0000     | 5.0000            | 0.0075          | -3563.7586         | NaN                    | NaN               | NaN               | 0.4306                          | 0.0583                        | 0.3838                        | 0.0564                      | 60.0000         |

## Tree stability
| tree               | cophenetic_corr | bootstrap_cophenetic_corr_mean | bootstrap_branch_jaccard_mean | bootstrap_branch_jaccard_sd |
| ------------------ | --------------- | ------------------------------ | ----------------------------- | --------------------------- |
| log1p              | 0.5624          | 0.5575                         | 0.1120                        | 0.0079                      |
| CLR + capture bias | 0.4157          | 0.3660                         | 0.1495                        | 0.0129                      |

## Geometry
| pairwise_distance_spearman_log_vs_clr |
| ------------------------------------- |
| -0.3140                               |

## Query mapping
Tested query sample: `RNAmatrix_HtanZ_S59.tsv`.
Query cells: `300`.
Genes used after capping: `3000`.
Neighborhood size: `k=30`.
Log1p uses gene-wise mean adjusted `log1p(counts)`.
CLR uses capture-bias corrected counts before centered log-ratio transform.
MNB is not a kNN method. It scores all reference labels once per cell and reports posterior confidence, score gap, entropy, and bootstrap agreement.

| metric | log1p_gene_centered | clr_bias_corrected | mnb_bias_corrected |
| ------ | ------------------- | ------------------ | ------------------ |
| assignment_confidence_mean | 0.4717 | 0.7079 | 0.8646 |
| margin_or_score_gap_mean | 0.0452 | 0.4750 | 7.2580 |
| entropy_mean | 1.0077 | 0.7812 | 0.3380 |
| bootstrap_stability_mean | 0.1120 | 0.1495 | 0.8956 |
| reference_30nn_homogeneity | 0.6060 | 0.6167 | n/a |
| query_tree_homogeneity | 0.4717 | 0.7079 | n/a |
| common_branch_height | 2856.3963 | 4008.0506 | n/a |
| unique_predictions | 2.0000 | 1.0000 | 4.0000 |

Metric notes:
- `assignment_confidence_mean`: 30-NN majority fraction for log1p and CLR; maximum posterior probability for MNB.
- `margin_or_score_gap_mean`: top-1 minus top-2 neighbor share for log1p and CLR; log-posterior score gap for MNB.
- `entropy_mean`: entropy of the 30-NN label distribution for log1p and CLR; entropy of the posterior label distribution for MNB.
- `bootstrap_stability_mean`: bootstrap branch Jaccard for the reference Ward trees in log1p and CLR; bootstrap prediction agreement for MNB.
- `reference_30nn_homogeneity`: fraction of the 30 nearest reference neighbors that share the assigned label.
- `query_tree_homogeneity`: same neighborhood purity, but evaluated against the reference Ward tree neighborhoods.
- `common_branch_height`: mean Ward-tree height of the lowest common ancestor of the 30 nearest reference neighbors.
- `unique_predictions`: number of distinct labels assigned by each method.
- `log1p` and `CLR` use 30-NN Euclidean neighborhoods in their transformed spaces.
- `MNB` is not a 30-NN procedure; it classifies directly against all reference labels once per cell.

## Refdata multiplets
| sample        | n_pseudobulks | multiplet_fraction | median_multiplet_score | median_lrt |
| ------------- | ------------- | ------------------ | ---------------------- | ---------- |
| BelleDN_raw   | 2210.0000     | 0.5747             | 1.0000                 | 82.7587    |
| Carpenter_raw | 1178.0000     | 0.4380             | 0.0000                 | 0.0000     |
| Chen_raw      | 794.0000      | 0.4987             | 0.9424                 | 3.6074     |
| Hajkarim_raw  | 1080.0000     | 0.5667             | 1.0000                 | 33.1916    |
| Hwang_raw     | 1156.0000     | 0.5407             | 1.0000                 | 20.0619    |
| Lin_raw       | 244.0000      | 0.5410             | 1.0000                 | 19.7755    |
| Peng_raw      | 782.0000      | 0.5742             | 1.0000                 | 48.9109    |
| Raghavan_raw  | 361.0000      | 0.5402             | 1.0000                 | 23.8870    |
| Steele_raw    | 644.0000      | 0.4783             | 0.7300                 | 1.2190     |
| Storrs_raw    | 624.0000      | 0.6122             | 1.0000                 | 86.6082    |
| Werba_raw     | 2426.0000     | 0.5627             | 1.0000                 | 60.6014    |
| Zhang_raw     | 720.0000      | 0.4167             | 0.0000                 | 0.0000     |
