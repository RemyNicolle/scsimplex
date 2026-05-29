#!/usr/bin/env python3
"""Benchmark scsimplex against the supplied benchmark assets and local cellstates outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scsimplex.pp import calibrate_capture_bias, clr_transform, multinomial_kmeans  # noqa: E402
from scsimplex.tl import detect_multiplets, map_multinomial_nb  # noqa: E402


@dataclass(slots=True)
class BenchmarkMatrix:
    """Gene-by-cell matrix loaded from a TSV benchmark file."""

    matrix: np.ndarray
    gene_names: np.ndarray
    cell_names: np.ndarray
    dataset_name: str


class SimpleAnnData:
    """Minimal AnnData-like object for the local benchmark runner."""

    def __init__(self, X: object, obs: pd.DataFrame, var: pd.DataFrame) -> None:
        self.X = X
        self.obs = obs
        self.var = var
        self.layers: dict[str, object] = {}
        self.uns: dict[str, object] = {}
        self.obsm: dict[str, object] = {}

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    def copy(self) -> "SimpleAnnData":
        copied = SimpleAnnData(
            X=self.X.copy() if hasattr(self.X, "copy") else self.X,
            obs=self.obs.copy(),
            var=self.var.copy(),
        )
        copied.layers = {key: value.copy() if hasattr(value, "copy") else value for key, value in self.layers.items()}
        copied.uns = dict(self.uns)
        copied.obsm = {key: value.copy() if hasattr(value, "copy") else value for key, value in self.obsm.items()}
        return copied


def _dense(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _load_gene_by_cell_tsv(path: Path, max_genes: int, max_cells: int) -> BenchmarkMatrix:
    usecols = list(range(min(max_cells, _count_cells(path)) + 1))
    df = pd.read_csv(path, sep="\t", index_col=0, nrows=max_genes, usecols=usecols)
    gene_names = df.index.astype(str).to_numpy()
    cell_names = df.columns.astype(str).to_numpy()
    matrix = df.to_numpy(dtype=np.float32, copy=False).T
    return BenchmarkMatrix(matrix=matrix, gene_names=gene_names, cell_names=cell_names, dataset_name=path.stem)


def _count_cells(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    return max(0, len(header) - 1)


def _count_genes(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def _make_adata(data: BenchmarkMatrix) -> SimpleAnnData:
    return SimpleAnnData(
        X=sp.csr_matrix(data.matrix),
        obs=pd.DataFrame(index=data.cell_names),
        var=pd.DataFrame(index=data.gene_names),
    )


def _sample_ground_truth(sample_name: str, benchmark_dir: Path, max_cells: int) -> Optional[pd.Series]:
    result_dir = benchmark_dir / "cellstateResults" / sample_name
    cluster_path = result_dir / "optimized_clusters.txt"
    cellid_path = result_dir / "CellID.txt"
    if not cluster_path.exists() or not cellid_path.exists():
        return None
    clusters = pd.read_csv(cluster_path, header=None, sep="\t").iloc[:max_cells, 0].astype(str).to_numpy()
    cell_ids = pd.read_csv(cellid_path, header=None, sep="\t").iloc[:max_cells, 0].astype(str).to_numpy()
    return pd.Series(clusters, index=cell_ids)


def _cluster_profiles(counts: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    profiles = np.vstack([(counts[labels == label].sum(axis=0) + 1e-12) for label in unique_labels])
    profiles /= profiles.sum(axis=1, keepdims=True)
    return unique_labels, profiles


def _mean_log_likelihood(counts: np.ndarray, labels: np.ndarray) -> float:
    unique_labels, profiles = _cluster_profiles(counts, labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    ll = 0.0
    for row, label in zip(counts, labels):
        ll += float(np.dot(row, np.log(profiles[label_to_idx[label]] + 1e-12)))
    return ll / counts.shape[0]


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size:
        raise ValueError("Label vectors must have the same size.")

    labels_a, inv_a = np.unique(a, return_inverse=True)
    labels_b, inv_b = np.unique(b, return_inverse=True)
    contingency = np.zeros((labels_a.size, labels_b.size), dtype=np.int64)
    np.add.at(contingency, (inv_a, inv_b), 1)

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) // 2

    sum_comb_c = comb2(contingency).sum()
    sum_comb_a = comb2(contingency.sum(axis=1)).sum()
    sum_comb_b = comb2(contingency.sum(axis=0)).sum()
    n = a.size
    total = n * (n - 1) // 2
    if total == 0:
        return 1.0
    expected = sum_comb_a * sum_comb_b / total
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 0.0
    return float((sum_comb_c - expected) / (max_index - expected))


def _normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    labels_a, inv_a = np.unique(a, return_inverse=True)
    labels_b, inv_b = np.unique(b, return_inverse=True)
    contingency = np.zeros((labels_a.size, labels_b.size), dtype=np.float64)
    np.add.at(contingency, (inv_a, inv_b), 1.0)
    contingency /= contingency.sum()
    pa = contingency.sum(axis=1)
    pb = contingency.sum(axis=0)
    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            pij = contingency[i, j]
            if pij > 0 and pa[i] > 0 and pb[j] > 0:
                mi += pij * math.log(pij / (pa[i] * pb[j]))
    ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    if ha == 0.0 or hb == 0.0:
        return 1.0
    return float(mi / math.sqrt(ha * hb))


def _majority_vote_nn(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int = 30) -> np.ndarray:
    tree = cKDTree(train_x)
    _, idx = tree.query(query_x, k=min(k, train_x.shape[0]))
    if idx.ndim == 1:
        idx = idx[:, np.newaxis]
    predictions = []
    for row in idx:
        labels, counts = np.unique(train_y[row], return_counts=True)
        predictions.append(labels[int(np.argmax(counts))])
    return np.asarray(predictions)


def _knn_homogeneity(features: np.ndarray, labels: np.ndarray, k: int = 30, n_rows: int = 50) -> float:
    tree = cKDTree(features)
    sample_count = min(n_rows, features.shape[0])
    _, idx = tree.query(features[:sample_count], k=min(k + 1, features.shape[0]))
    if idx.ndim == 1:
        idx = idx[:, np.newaxis]
    scores = []
    for row_idx, neighbours in enumerate(idx):
        neighbours = neighbours[neighbours != row_idx][:k]
        if neighbours.size == 0:
            continue
        scores.append(float(np.mean(labels[neighbours] == labels[row_idx])))
    return float(np.mean(scores)) if scores else float("nan")


def _log1p_features(matrix: np.ndarray) -> np.ndarray:
    return np.log1p(matrix)


def _gene_centered_log1p(matrix: np.ndarray) -> np.ndarray:
    features = np.log1p(matrix)
    features = features - features.mean(axis=0, keepdims=True)
    return features


def _label_distribution_entropy(labels: np.ndarray) -> np.ndarray:
    tree = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(tree)}
    encoded = np.asarray([label_to_index[label] for label in labels], dtype=int)
    return encoded


def _knn_prediction_stats(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(train_x)
    _, idx = tree.query(query_x, k=min(k, train_x.shape[0]))
    if idx.ndim == 1:
        idx = idx[:, np.newaxis]
    predictions: list[str] = []
    majorities: list[float] = []
    margins: list[float] = []
    entropies: list[float] = []
    for row in idx:
        labels, counts = np.unique(train_y[row], return_counts=True)
        proportions = counts / counts.sum()
        order = np.argsort(counts)[::-1]
        top1 = float(proportions[order[0]])
        top2 = float(proportions[order[1]]) if order.size > 1 else 0.0
        predictions.append(str(labels[int(order[0])]))
        majorities.append(top1)
        margins.append(top1 - top2)
        entropies.append(float(-(proportions * np.log(proportions + 1e-12)).sum()))
    return np.asarray(predictions), np.asarray(majorities), np.asarray(margins), np.asarray(entropies)


def _clr_features(adata: SimpleAnnData) -> np.ndarray:
    clr_transform(adata)
    return _dense(adata.layers["X_clr"])


def _bootstrap_tree_stability(features: np.ndarray, n_boot: int, sample_fraction: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_cells = features.shape[0]
    sample_size = max(3, int(round(n_cells * sample_fraction)))
    corrs = []
    for _ in range(n_boot):
        idx = rng.choice(n_cells, size=sample_size, replace=True)
        sample = features[idx]
        distances = pdist(sample, metric="euclidean")
        if distances.size == 0:
            continue
        tree = linkage(distances, method="ward")
        corr, _ = cophenet(tree, distances)
        corrs.append(float(corr))
    if len(corrs) == 0:
        return float("nan"), float("nan")
    return float(np.mean(corrs)), float(np.std(corrs, ddof=1) if len(corrs) > 1 else 0.0)


def _tree_summary(features: np.ndarray, bootstrap: int, seed: int) -> dict[str, float]:
    tree = linkage(pdist(features, metric="euclidean"), method="ward")
    distances = pdist(features, metric="euclidean")
    cophenetic_corr, _ = cophenet(tree, distances)
    boot_coph, branch_jaccard_mean, branch_jaccard_sd = _bootstrap_branch_jaccard(
        features,
        n_boot=bootstrap,
        sample_fraction=0.2,
        seed=seed,
    )
    return {
        "cophenetic_corr": float(cophenetic_corr),
        "bootstrap_cophenetic_corr_mean": float(boot_coph),
        "bootstrap_branch_jaccard_mean": float(branch_jaccard_mean),
        "bootstrap_branch_jaccard_sd": float(branch_jaccard_sd),
    }


def _sampled_pairwise_spearman(features_a: np.ndarray, features_b: np.ndarray, seed: int, n_rows: int = 400) -> float:
    rng = np.random.default_rng(seed)
    n_cells = min(n_rows, features_a.shape[0], features_b.shape[0])
    if n_cells < 3:
        return float("nan")
    idx = rng.choice(features_a.shape[0], size=n_cells, replace=False)
    d_a = pdist(features_a[idx], metric="euclidean")
    d_b = pdist(features_b[idx], metric="euclidean")
    corr = spearmanr(d_a, d_b).correlation
    return float(corr) if corr is not None else float("nan")


def _score_gap_and_entropy(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.ndim != 2 or scores.shape[1] < 2:
        zeros = np.zeros(scores.shape[0], dtype=float)
        return zeros, zeros, zeros
    sorted_scores = np.sort(scores, axis=1)
    gaps = sorted_scores[:, -1] - sorted_scores[:, -2]
    centered = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(centered)
    probs /= probs.sum(axis=1, keepdims=True)
    max_prob = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    return gaps, entropy, max_prob


def _mnb_bootstrap_agreement(
    query: SimpleAnnData,
    reference: SimpleAnnData,
    reference_cluster_key: str,
    baseline_predictions: np.ndarray,
    n_boot: int = 3,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    agreements = []
    ref_dense = _dense(reference.X)
    ref_labels = reference.obs[reference_cluster_key].astype(str).to_numpy()
    for _ in range(n_boot):
        boot_idx = rng.choice(ref_dense.shape[0], size=ref_dense.shape[0], replace=True)
        boot_ref = SimpleAnnData(
            X=sp.csr_matrix(ref_dense[boot_idx]),
            obs=reference.obs.iloc[boot_idx].copy(),
            var=reference.var.copy(),
        )
        if "capture_bias_beta" in query.uns:
            boot_query = query.copy()
            predicted = map_multinomial_nb(
                boot_query,
                boot_ref,
                reference_cluster_key=reference_cluster_key,
                use_bias_correction=True,
            )
        else:
            boot_query = query.copy()
            predicted = map_multinomial_nb(
                boot_query,
                boot_ref,
                reference_cluster_key=reference_cluster_key,
                use_bias_correction=False,
            )
        agreements.append(float(np.mean(predicted.astype(str) == baseline_predictions.astype(str))))
    return (
        float(np.mean(agreements)) if agreements else float("nan"),
        float(np.std(agreements, ddof=1)) if len(agreements) > 1 else 0.0,
    )


def _mapping_summary_table(metrics: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "metric": "assignment_confidence_mean",
            "log1p_gene_centered": metrics["query_knn_log1p_majority_mean"],
            "clr_bias_corrected": metrics["query_knn_clr_majority_mean"],
            "mnb_bias_corrected": metrics["query_mnb_bias_max_prob_mean"],
        },
        {
            "metric": "margin_or_score_gap_mean",
            "log1p_gene_centered": metrics["query_knn_log1p_margin_mean"],
            "clr_bias_corrected": metrics["query_knn_clr_margin_mean"],
            "mnb_bias_corrected": metrics["query_mnb_bias_score_gap_mean"],
        },
        {
            "metric": "entropy_mean",
            "log1p_gene_centered": metrics["query_knn_log1p_entropy_mean"],
            "clr_bias_corrected": metrics["query_knn_clr_entropy_mean"],
            "mnb_bias_corrected": metrics["query_mnb_bias_entropy_mean"],
        },
        {
            "metric": "bootstrap_stability_mean",
            "log1p_gene_centered": metrics["log1p_bootstrap_branch_jaccard_mean"],
            "clr_bias_corrected": metrics["clr_bootstrap_branch_jaccard_mean"],
            "mnb_bias_corrected": metrics["mnb_bootstrap_agreement_mean"],
        },
        {
            "metric": "reference_30nn_homogeneity",
            "log1p_gene_centered": metrics["reference_30nn_homogeneity_log1p"],
            "clr_bias_corrected": metrics["reference_30nn_homogeneity_clr"],
            "mnb_bias_corrected": np.nan,
        },
        {
            "metric": "query_tree_homogeneity",
            "log1p_gene_centered": metrics["query_knn_tree_homogeneity_log1p"],
            "clr_bias_corrected": metrics["query_knn_tree_homogeneity_clr"],
            "mnb_bias_corrected": np.nan,
        },
        {
            "metric": "common_branch_height",
            "log1p_gene_centered": metrics["query_common_branch_height_log1p"],
            "clr_bias_corrected": metrics["query_common_branch_height_clr"],
            "mnb_bias_corrected": np.nan,
        },
        {
            "metric": "unique_predictions",
            "log1p_gene_centered": metrics["query_knn_log1p_unique"],
            "clr_bias_corrected": metrics["query_knn_clr_unique"],
            "mnb_bias_corrected": metrics["query_mnb_bias_unique"],
        },
    ]
    return pd.DataFrame(rows)


def _mapping_metric_notes() -> list[str]:
    return [
        "- `assignment_confidence_mean`: 30-NN majority fraction for log1p and CLR; maximum posterior probability for MNB.",
        "- `margin_or_score_gap_mean`: top-1 minus top-2 neighbor share for log1p and CLR; log-posterior score gap for MNB.",
        "- `entropy_mean`: entropy of the 30-NN label distribution for log1p and CLR; entropy of the posterior label distribution for MNB.",
        "- `bootstrap_stability_mean`: mean bootstrap branch Jaccard for log1p and CLR trees; bootstrap prediction agreement for MNB.",
        "- `reference_30nn_homogeneity`: fraction of the 30 nearest reference neighbors that share the assigned label.",
        "- `query_tree_homogeneity`: same neighborhood purity, but evaluated against the reference Ward tree neighborhoods.",
        "- `common_branch_height`: mean Ward-tree height of the lowest common ancestor of the 30 nearest reference neighbors.",
        "- `unique_predictions`: number of distinct labels assigned by each method.",
    ]


def _build_reference(ref_dir: Path, max_genes: int, max_cells: int) -> SimpleAnnData:
    annot = pd.read_csv(ref_dir / "clusters_annot_k7.tsv", sep="\t")
    label_map = annot.set_index("pseudo_bulk")["annotation"].to_dict()
    matrix_parts = []
    obs_frames = []
    gene_names: Optional[np.ndarray] = None

    for path in sorted(ref_dir.glob("*_raw.txt")):
        data = _load_gene_by_cell_tsv(path, max_genes=max_genes, max_cells=max_cells)
        if gene_names is None:
            gene_names = data.gene_names
        elif not np.array_equal(gene_names, data.gene_names):
            common = np.intersect1d(gene_names, data.gene_names, assume_unique=False)
            if common.size == 0:
                raise ValueError("Reference files do not share a common gene set.")
            gene_names = common
        matrix_parts.append(data.matrix)
        obs_frames.append(pd.DataFrame(index=data.cell_names))

    combined = np.concatenate(matrix_parts, axis=0)
    obs = pd.concat(obs_frames, axis=0)
    obs["reference_state"] = [label_map.get(name, "unknown") for name in obs.index]
    var = pd.DataFrame(index=gene_names if gene_names is not None else np.array([], dtype=str))
    return SimpleAnnData(X=sp.csr_matrix(combined), obs=obs, var=var)


def _load_refdata_datasets(ref_dir: Path, max_genes: int) -> list[SimpleAnnData]:
    annot = pd.read_csv(ref_dir / "clusters_annot_k7.tsv", sep="\t")
    pseudo_to_cluster = annot.set_index("pseudo_bulk")["cluster"].to_dict()
    pseudo_to_annotation = annot.set_index("pseudo_bulk")["annotation"].to_dict()
    datasets: list[SimpleAnnData] = []

    for path in sorted(ref_dir.glob("*_raw.txt")):
        data = _load_gene_by_cell_tsv(path, max_genes=max_genes, max_cells=_count_cells(path))
        obs = pd.DataFrame(index=data.cell_names)
        obs["cluster"] = [pseudo_to_cluster.get(name, -1) for name in data.cell_names]
        obs["annotation"] = [pseudo_to_annotation.get(name, "unknown") for name in data.cell_names]
        obs["dataset"] = path.stem
        datasets.append(SimpleAnnData(X=sp.csr_matrix(data.matrix), obs=obs, var=pd.DataFrame(index=data.gene_names)))
    return datasets


def _run_refdata_multiplets(ref_dir: Path, max_genes: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    annot = pd.read_csv(ref_dir / "clusters_annot_k7.tsv", sep="\t")
    pseudo_to_annotation = annot.set_index("pseudo_bulk")["annotation"].to_dict()
    for path in sorted(ref_dir.glob("*_raw.txt")):
        data = _load_gene_by_cell_tsv(path, max_genes=max_genes, max_cells=_count_cells(path))
        adata = _make_adata(data)
        adata.obs["annotation"] = [pseudo_to_annotation.get(name, "unknown") for name in data.cell_names]
        detect_multiplets(adata, cluster_key="annotation", alpha=0.05)
        rows.append(
            {
                "sample": path.stem,
                "n_pseudobulks": float(adata.X.shape[0]),
                "multiplet_fraction": float(np.mean(adata.obs["is_multiplet"].to_numpy())),
                "median_multiplet_score": float(np.median(adata.obs["multiplet_score"].to_numpy())),
                "median_lrt": float(np.median(adata.obs["multiplet_lrt"].to_numpy())),
            }
        )
    return pd.DataFrame(rows)


def _concat_datasets(datasets: list[SimpleAnnData]) -> SimpleAnnData:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required.")
    gene_names = datasets[0].var.index.astype(str).to_numpy()
    matrices = []
    obs_frames = []
    for dataset in datasets:
        if not np.array_equal(gene_names, dataset.var.index.astype(str).to_numpy()):
            raise ValueError("All refdata datasets must share the same gene axis after slicing.")
        matrices.append(_dense(dataset.X))
        obs_frames.append(dataset.obs.copy())
    combined = np.concatenate(matrices, axis=0)
    obs = pd.concat(obs_frames, axis=0)
    return SimpleAnnData(X=sp.csr_matrix(combined), obs=obs, var=pd.DataFrame(index=gene_names))


def _compare_mapping(reference: SimpleAnnData, query: SimpleAnnData, anchor_label: str) -> dict[str, float]:
    ref_dense = _dense(reference.X)
    query_dense = _dense(query.X)
    ref_labels = reference.obs["reference_state"].astype(str).to_numpy()

    ref_log = _gene_centered_log1p(ref_dense)
    query_log = _gene_centered_log1p(query_dense)
    ref_clr = _clr_features(reference.copy())
    query_clr = _clr_features(query.copy())

    knn_log, knn_log_majority, knn_log_margin, knn_log_entropy = _knn_prediction_stats(ref_log, ref_labels, query_log, k=30)
    knn_clr, knn_clr_majority, knn_clr_margin, knn_clr_entropy = _knn_prediction_stats(ref_clr, ref_labels, query_clr, k=30)
    tree_log = _query_tree_metrics(ref_log, ref_labels, query_log, k=30)
    tree_clr = _query_tree_metrics(ref_clr, ref_labels, query_clr, k=30)

    provisional = knn_log.copy()
    query.obs["reference_state"] = provisional
    shared_labels = [label for label in pd.Series(ref_labels).value_counts().index if label in set(provisional)]
    chosen_anchor = anchor_label if anchor_label in set(provisional) else (shared_labels[0] if shared_labels else None)
    if chosen_anchor is not None:
        calibrate_capture_bias([reference, query], anchor_cluster_obs_key="reference_state", anchor_cluster_name=chosen_anchor)
        query_bias = query.copy()
        mnb_bias = map_multinomial_nb(query_bias, reference, reference_cluster_key="reference_state", use_bias_correction=True)
        bias_gap, bias_entropy, bias_max_prob = _score_gap_and_entropy(query_bias.obsm["predicted_cell_state_scores"])
        mnb_bootstrap_mean, mnb_bootstrap_sd = _mnb_bootstrap_agreement(
            query,
            reference,
            "reference_state",
            baseline_predictions=mnb_bias,
            n_boot=3,
            seed=0,
        )
    else:
        mnb_bias = map_multinomial_nb(query, reference, reference_cluster_key="reference_state", use_bias_correction=False)
        bias_gap, bias_entropy, bias_max_prob = _score_gap_and_entropy(query.obsm["predicted_cell_state_scores"])
        mnb_bootstrap_mean, mnb_bootstrap_sd = _mnb_bootstrap_agreement(
            query,
            reference,
            "reference_state",
            baseline_predictions=mnb_bias,
            n_boot=3,
            seed=0,
        )
    query_nobias = query.copy()
    mnb_nobias = map_multinomial_nb(query_nobias, reference, reference_cluster_key="reference_state", use_bias_correction=False)
    nobias_gap, nobias_entropy, nobias_max_prob = _score_gap_and_entropy(query_nobias.obsm["predicted_cell_state_scores"])

    hom_log = _knn_homogeneity(ref_log, ref_labels, k=30)
    hom_clr = _knn_homogeneity(ref_clr, ref_labels, k=30)

    return {
        "query_knn_log1p_unique": float(np.unique(knn_log).size),
        "query_knn_clr_unique": float(np.unique(knn_clr).size),
        "query_knn_log1p_majority_mean": float(np.mean(knn_log_majority)),
        "query_knn_clr_majority_mean": float(np.mean(knn_clr_majority)),
        "query_knn_log1p_margin_mean": float(np.mean(knn_log_margin)),
        "query_knn_clr_margin_mean": float(np.mean(knn_clr_margin)),
        "query_knn_log1p_entropy_mean": float(np.mean(knn_log_entropy)),
        "query_knn_clr_entropy_mean": float(np.mean(knn_clr_entropy)),
        "query_knn_log1p_vs_clr_agreement": float(np.mean(knn_log.astype(str) == knn_clr.astype(str))),
        "query_mnb_bias_unique": float(np.unique(mnb_bias).size),
        "query_mnb_nobias_unique": float(np.unique(mnb_nobias).size),
        "query_bias_agreement": float(np.mean(mnb_bias.astype(str) == mnb_nobias.astype(str))),
        "query_mnb_bias_max_prob_mean": float(np.mean(bias_max_prob)),
        "query_mnb_bias_score_gap_mean": float(np.mean(bias_gap)),
        "query_mnb_nobias_score_gap_mean": float(np.mean(nobias_gap)),
        "query_mnb_bias_entropy_mean": float(np.mean(bias_entropy)),
        "query_mnb_nobias_entropy_mean": float(np.mean(nobias_entropy)),
        "query_mnb_nobias_max_prob_mean": float(np.mean(nobias_max_prob)),
        "mnb_bootstrap_agreement_mean": mnb_bootstrap_mean,
        "mnb_bootstrap_agreement_sd": mnb_bootstrap_sd,
        "reference_30nn_homogeneity_log1p": hom_log,
        "reference_30nn_homogeneity_clr": hom_clr,
        "query_knn_tree_homogeneity_log1p": tree_log["knn_homogeneity"],
        "query_knn_tree_homogeneity_clr": tree_clr["knn_homogeneity"],
        "query_common_branch_height_log1p": tree_log["common_branch_height"],
        "query_common_branch_height_clr": tree_clr["common_branch_height"],
    }


def _benchmark_sample(matrix_path: Path, benchmark_dir: Path, max_genes: int, max_cells: int, bootstrap: int) -> dict[str, float]:
    data = _load_gene_by_cell_tsv(matrix_path, max_genes=max_genes, max_cells=max_cells)
    adata = _make_adata(data)
    counts = _dense(adata.X)

    ground_truth = _sample_ground_truth(data.dataset_name, benchmark_dir, max_cells=max_cells)
    truth_series = ground_truth.reindex(data.cell_names) if ground_truth is not None else None
    valid_mask = truth_series.notna().to_numpy() if truth_series is not None else np.zeros(counts.shape[0], dtype=bool)

    t0 = time.perf_counter()
    multinomial_kmeans(adata, target_metacell_size=5)
    kmeans_time = time.perf_counter() - t0
    kmeans_labels = adata.obs["multinomial_kmeans_cluster"].astype(str).to_numpy()

    pseudobulk = np.vstack([counts[kmeans_labels == label].sum(axis=0) for label in np.unique(kmeans_labels)])
    pseudobulk_labels = np.unique(kmeans_labels)

    data_ll = _mean_log_likelihood(counts, kmeans_labels)
    if truth_series is not None and valid_mask.any():
        truth_labels = truth_series[valid_mask].astype(str).to_numpy()
        truth_ll = _mean_log_likelihood(counts[valid_mask], truth_labels)
        ari = _adjusted_rand_index(kmeans_labels[valid_mask], truth_labels)
        nmi = _normalized_mutual_info(kmeans_labels[valid_mask], truth_labels)
    else:
        truth_ll = float("nan")
        ari = float("nan")
        nmi = float("nan")

    log_features = _gene_centered_log1p(counts)
    clr_adata = adata.copy()
    clr_transform(clr_adata)
    clr_features = _dense(clr_adata.layers["X_clr"])
    log_boot_mean, log_boot_sd = _bootstrap_tree_stability(log_features, n_boot=bootstrap, sample_fraction=0.8, seed=0)
    clr_boot_mean, clr_boot_sd = _bootstrap_tree_stability(clr_features, n_boot=bootstrap, sample_fraction=0.8, seed=1)

    return {
        "sample": data.dataset_name,
        "n_cells": float(counts.shape[0]),
        "n_genes": float(counts.shape[1]),
        "n_metacells": float(np.unique(kmeans_labels).size),
        "compression_ratio": float(counts.shape[0] / np.unique(kmeans_labels).size),
        "kmeans_time_sec": float(kmeans_time),
        "mean_loglik_kmeans": float(data_ll),
        "mean_loglik_cellstates": float(truth_ll),
        "ari_vs_cellstates": float(ari),
        "nmi_vs_cellstates": float(nmi),
        "bootstrap_cophenetic_log1p_mean": float(log_boot_mean),
        "bootstrap_cophenetic_log1p_sd": float(log_boot_sd),
        "bootstrap_cophenetic_clr_mean": float(clr_boot_mean),
        "bootstrap_cophenetic_clr_sd": float(clr_boot_sd),
        "pseudobulk_rows": float(pseudobulk.shape[0]),
    }


def _format_table(df: pd.DataFrame) -> str:
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: f"{x:0.4f}" if pd.notna(x) else "NaN")
    headers = list(frame.columns)
    rows = [headers] + frame.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([head, sep, *body])


def _leaf_ancestors(linkage_matrix: np.ndarray, n_leaves: int) -> tuple[list[list[int]], np.ndarray]:
    parent = np.full(2 * n_leaves - 1, -1, dtype=int)
    heights = np.zeros(2 * n_leaves - 1, dtype=float)
    for i, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node = n_leaves + i
        parent[left] = node
        parent[right] = node
        heights[node] = float(row[2])
    ancestor_paths: list[list[int]] = []
    for leaf in range(n_leaves):
        path = [leaf]
        node = leaf
        while parent[node] != -1:
            node = int(parent[node])
            path.append(node)
        ancestor_paths.append(path)
    return ancestor_paths, heights


def _lca_height(ancestor_paths: list[list[int]], heights: np.ndarray, leaf_indices: np.ndarray) -> float:
    common = set(ancestor_paths[int(leaf_indices[0])])
    for leaf in leaf_indices[1:]:
        common.intersection_update(ancestor_paths[int(leaf)])
        if not common:
            return 0.0
    return float(max(heights[node] for node in common))


def _bootstrap_branch_jaccard(
    features: np.ndarray,
    n_boot: int,
    sample_fraction: float,
    seed: int,
    min_branch_size: int = 100,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_cells = features.shape[0]
    sample_size = max(min_branch_size + 1, int(round(n_cells * sample_fraction)))
    sample_size = min(sample_size, 1000, n_cells)
    reference_idx = rng.choice(n_cells, size=sample_size, replace=True)
    reference_sample = features[reference_idx]
    reference_tree = linkage(pdist(reference_sample, metric="euclidean"), method="ward")
    reference_k = max(2, reference_sample.shape[0] // min_branch_size)
    reference_clusters = fcluster(reference_tree, t=reference_k, criterion="maxclust")
    reference_sets = [set(np.where(reference_clusters == label)[0].tolist()) for label in np.unique(reference_clusters)]
    branch_scores: list[float] = []
    cophs: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n_cells, size=sample_size, replace=True)
        sample = features[idx]
        tree = linkage(pdist(sample, metric="euclidean"), method="ward")
        distances = pdist(sample, metric="euclidean")
        corr, _ = cophenet(tree, distances)
        cophs.append(float(corr))
        k = max(2, sample.shape[0] // min_branch_size)
        clusters = fcluster(tree, t=k, criterion="maxclust")
        boot_sets = [set(np.where(clusters == label)[0].tolist()) for label in np.unique(clusters)]
        best_jaccard = []
        for ref_set in reference_sets:
            if len(ref_set) < min_branch_size:
                continue
            ref_size = float(len(ref_set))
            best = 0.0
            for boot_set in boot_sets:
                inter = len(ref_set & boot_set)
                union = len(ref_set | boot_set)
                if union > 0:
                    best = max(best, inter / union)
            best_jaccard.append(best)
        if best_jaccard:
            branch_scores.append(float(np.mean(best_jaccard)))
    return (
        float(np.mean(cophs)) if cophs else float("nan"),
        float(np.mean(branch_scores)) if branch_scores else float("nan"),
        float(np.std(branch_scores, ddof=1)) if len(branch_scores) > 1 else 0.0,
    )


def _query_tree_metrics(reference_features: np.ndarray, reference_labels: np.ndarray, query_features: np.ndarray, k: int = 30) -> dict[str, float]:
    tree = linkage(pdist(reference_features, metric="euclidean"), method="ward")
    ancestor_paths, heights = _leaf_ancestors(tree, reference_features.shape[0])
    knn = cKDTree(reference_features)
    _, idx = knn.query(query_features, k=min(k, reference_features.shape[0]))
    if idx.ndim == 1:
        idx = idx[:, np.newaxis]
    homogeneity_scores = []
    lca_heights = []
    for row in idx:
        labels = reference_labels[row]
        majority = np.unique(labels, return_counts=True)[1].max() / float(labels.size)
        homogeneity_scores.append(float(majority))
        lca_heights.append(_lca_height(ancestor_paths, heights, row))
    return {
        "knn_homogeneity": float(np.mean(homogeneity_scores)) if homogeneity_scores else float("nan"),
        "common_branch_height": float(np.mean(lca_heights)) if lca_heights else float("nan"),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("scripts/Benchmark"))
    parser.add_argument("--ref-dir", type=Path, default=Path("scripts/refdata"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_reports"))
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--max-genes", type=int, default=500)
    parser.add_argument("--max-cells", type=int, default=800)
    parser.add_argument("--reference-max-cells", type=int, default=1200)
    parser.add_argument("--bootstrap", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = args.samples
    if sample_files is None:
        sample_files = [path.name for path in sorted(args.benchmark_dir.glob("RNAmatrix_*.tsv"))]

    sample_rows = []
    for sample in sample_files:
        sample_rows.append(
            _benchmark_sample(
                args.benchmark_dir / sample,
                args.benchmark_dir,
                max_genes=args.max_genes,
                max_cells=args.max_cells,
                bootstrap=args.bootstrap,
            )
        )

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(args.output_dir / "sample_metrics.csv", index=False)

    reference = _build_reference(args.ref_dir, max_genes=args.max_genes, max_cells=args.reference_max_cells)
    query = _make_adata(_load_gene_by_cell_tsv(args.benchmark_dir / "RNAmatrix_HtanZ_S59.tsv", max_genes=args.max_genes, max_cells=args.max_cells))
    anchor_candidates = reference.obs["reference_state"].astype(str).value_counts()
    anchor_label = anchor_candidates.index[0] if not anchor_candidates.empty else "epithelial"
    mapping_metrics = _compare_mapping(reference, query, anchor_label=anchor_label)

    ref_datasets = _load_refdata_datasets(args.ref_dir, max_genes=args.max_genes)
    ref_log_parts: list[np.ndarray] = []
    for dataset in ref_datasets:
        ref_log_parts.append(_gene_centered_log1p(_dense(dataset.X)))
    ref_log = np.concatenate(ref_log_parts, axis=0)

    annotation_sets = [set(dataset.obs["annotation"].astype(str).tolist()) for dataset in ref_datasets]
    common_annotations = set.intersection(*annotation_sets) if annotation_sets else set()
    if common_annotations:
        anchor_candidates = pd.Series(
            np.concatenate([dataset.obs["annotation"].astype(str).to_numpy() for dataset in ref_datasets], axis=0)
        )
        common_anchor = anchor_candidates[anchor_candidates.isin(common_annotations)].value_counts().index[0]
    else:
        anchor_candidates = pd.Series(
            np.concatenate([dataset.obs["annotation"].astype(str).to_numpy() for dataset in ref_datasets], axis=0)
        )
        common_anchor = anchor_candidates.value_counts().index[0]
    calibrate_capture_bias(ref_datasets, anchor_cluster_obs_key="annotation", anchor_cluster_name=common_anchor)
    for dataset in ref_datasets:
        beta = np.asarray(dataset.uns["capture_bias_beta"], dtype=float)
        corrected = _dense(dataset.X) * beta[np.newaxis, :]
        dataset.layers["capture_bias_corrected"] = corrected
        clr_transform(dataset, layer="capture_bias_corrected")
    ref_clr = np.concatenate([_dense(dataset.layers["X_clr"]) for dataset in ref_datasets], axis=0)
    tree_log_metrics = _tree_summary(ref_log, bootstrap=args.bootstrap, seed=0)
    tree_clr_metrics = _tree_summary(ref_clr, bootstrap=args.bootstrap, seed=1)
    tree_geometry_metrics = {
        "pairwise_distance_spearman_log_vs_clr": _sampled_pairwise_spearman(ref_log, ref_clr, seed=0)
    }
    mapping_metrics.update(
        {
            "log1p_bootstrap_branch_jaccard_mean": tree_log_metrics["bootstrap_branch_jaccard_mean"],
            "clr_bootstrap_branch_jaccard_mean": tree_clr_metrics["bootstrap_branch_jaccard_mean"],
            "log1p_bootstrap_branch_jaccard_sd": tree_log_metrics["bootstrap_branch_jaccard_sd"],
            "clr_bootstrap_branch_jaccard_sd": tree_clr_metrics["bootstrap_branch_jaccard_sd"],
        }
    )

    multiplet_df = _run_refdata_multiplets(args.ref_dir, max_genes=args.max_genes)
    multiplet_df.to_csv(args.output_dir / "refdata_multiplets.csv", index=False)

    summary = {
        "sample_metrics": sample_rows,
        "mapping_metrics": mapping_metrics,
        "tree_metrics_log1p": tree_log_metrics,
        "tree_metrics_clr": tree_clr_metrics,
        "tree_geometry_metrics": tree_geometry_metrics,
        "refdata_multiplets": multiplet_df.to_dict(orient="records"),
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    mapping_df = _mapping_summary_table(mapping_metrics)
    report_lines = [
        "# scsimplex benchmark summary",
        "",
        "## Sample metrics",
        _format_table(sample_df),
        "",
        "## Tree stability",
        _format_table(pd.DataFrame([
            {"tree": "log1p", **tree_log_metrics},
            {"tree": "CLR + capture bias", **tree_clr_metrics},
        ])),
        "",
        "## Geometry",
        _format_table(pd.DataFrame([tree_geometry_metrics])),
        "",
        "## Query mapping",
        "Tested query sample: `RNAmatrix_HtanZ_S59.tsv`.",
        f"Query cells: `{query.X.shape[0]}`.",
        f"Genes used after capping: `{query.X.shape[1]}`.",
        "Neighborhood size: `k=30`.",
        "Log1p uses gene-wise mean adjusted `log1p(counts)`.",
        "CLR uses capture-bias corrected counts before centered log-ratio transform.",
        "MNB is not a kNN method; it scores all reference labels once per cell and reports posterior confidence, score gap, entropy, and bootstrap agreement.",
        _format_table(mapping_df),
        "",
        "Metric notes:",
        *(_mapping_metric_notes()),
        "",
        "## Refdata multiplets",
        _format_table(multiplet_df),
    ]
    (args.output_dir / "benchmark_summary.md").write_text("\n".join(report_lines) + "\n")

    print(report_lines[0])
    print(report_lines[2])
    print(_format_table(sample_df))
    print()
    print(report_lines[5])
    print(_format_table(pd.DataFrame([
        {"tree": "log1p", **tree_log_metrics},
        {"tree": "CLR + capture bias", **tree_clr_metrics},
    ])))
    print()
    print(report_lines[8])
    print(_format_table(pd.DataFrame([tree_geometry_metrics])))
    print()
    print(report_lines[10])
    print("\n".join(report_lines[11:20]))
    print()
    print(report_lines[21])
    print(_format_table(multiplet_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
