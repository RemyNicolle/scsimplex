# scsimplex

Single-cell utilities built around multinomial and compositional geometry.

## Install

```bash
pip install .
```

For local development:

```bash
pip install -e ".[dev,faiss]"
```

## Use

### Main preprocessing

```python
from scsimplex.pp import calibrate_capture_bias, clr_transform

# Learn a reference from one or more datasets.
calibrated_simplex, reference_calibration = calibrate_capture_bias(adatas)

# Apply that reference to one new dataset.
query_simplex = calibrate_capture_bias(query_adata, reference_calibration=reference_calibration)

# CLR from counts or simplex.
clr_transform(query_adata, layer="capture_bias_corrected", out_layer="X_clr")
```

`bayesian_impute_pseudocounts`
- input: count matrix or AnnData count layer
- output: simplex matrix or stored simplex layer
- use when you have raw counts

`calibrate_capture_bias`
- input: counts or simplex, as matrices or AnnData
- not accepted: CLR
- list mode: learn one reference from several datasets
- single mode: apply an existing reference to one dataset
- output: calibrated simplex

`clr_transform`
- input: counts or simplex in AnnData
- not accepted: CLR as input
- output: CLR coordinates in a layer

### Pairwise-intersection distances

```python
from scsimplex.pp import pairwise_intersection_aitchison_distance_matrix

result = pairwise_intersection_aitchison_distance_matrix(adatas)
distances = result.distance_matrix
diagnostics = result.diagnostics
```

`pairwise_intersection_aitchison_distance_matrix`
- input: list of datasets, each as counts or simplex, as matrices or AnnData
- not accepted: CLR
- if gene names are present, pairwise intersections are used automatically
- if gene names are absent, all matrices must already have the same column order and width
- output: a global precomputed distance matrix plus diagnostics

Simple clustering from the distance matrix:

```python
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

result = pairwise_intersection_aitchison_distance_matrix(adatas)
tree = linkage(squareform(result.distance_matrix), method="average")
labels = fcluster(tree, t=10, criterion="maxclust")
```

Ward does not take a precomputed distance matrix directly. To use Ward, embed first:

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

embedding = MDS(n_components=20, dissimilarity="precomputed", random_state=0).fit_transform(
    result.distance_matrix
)
labels = AgglomerativeClustering(n_clusters=10, linkage="ward").fit_predict(embedding)
```

### Cell-tree visualisation

`celltreeviz` is vendored in `src/celltreeviz` and installed with the package.

```bash
celltreeviz path/to/tree.npy --leaf-data path/to/leaf_table.tsv --output tree.html
```

### Benchmark

```bash
python analysis/scripts/run_benchmark.py --help
```

Benchmark inputs live in `analysis/Benchmark/` and reference pseudobulks live in `analysis/refdata/`.
