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

### CLR with pseudocount imputation and capture-bias correction

```python
import anndata as ad
from scsimplex.pp import calibrate_capture_bias, bayesian_impute_pseudocounts, clr_transform

# `adatas` is a list of AnnData objects with a shared anchor cluster annotation.
calibrate_capture_bias(adatas, anchor_cluster_obs_key="cell_type", anchor_cluster_name="your_anchor")

for adata in adatas:
    bayesian_impute_pseudocounts(adata, out_layer="X_imputed")
    clr_transform(adata, layer="X_imputed", out_layer="X_clr")
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
