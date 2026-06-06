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
from scsimplex.pp import calibrate_capture_bias, clr_transform

# Learn a reference calibration from one or more datasets.
# Raw counts are converted to simplex automatically when needed.
calibrated_simplex, reference_calibration = calibrate_capture_bias(adatas)

# Apply the learned reference to a new dataset later on.
query_simplex = calibrate_capture_bias(query_adata, reference_calibration=reference_calibration)

for adata in adatas:
    clr_transform(adata, layer="capture_bias_corrected", out_layer="X_clr")
```

Dataset-center calibration assumes that differences between dataset compositional centers are
technical capture effects. If dataset-average biology genuinely differs, this calibration will
remove part of that biological difference.

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
