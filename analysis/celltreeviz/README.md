# celltreeviz

`celltreeviz` builds an interactive web visualisation for cell hierarchy trees:

- a central dendrogram or graph view;
- node and branch size/color driven by node-level metadata;
- optional hiding of nodes missing from the node metadata;
- a ComplexHeatmap-style leaf-aligned heatmap;
- continuous color scales with lower/upper clipping and optional midpoint;
- categorical color dictionaries that can be stored as YAML.

The library has one main entry point: `celltreeviz.visualize()`.

## Quick Start

```python
from celltreeviz import visualize

visualize(
    "dataex/Z.npy",
    node_data="dataex/clades_stability.parquet",
    leaf_data="dataex/leafdata.parquet",
    output="out/celltreeviz_example.html",
    node_size="stability",
    node_color="annotation",
    edge_size="stability",
    edge_color="annotation",
)

```

Open the generated HTML file in a browser. The file is self-contained by
default, including Plotly JavaScript.

You can also use the CLI:

```bash
celltreeviz dataex/Z.npy \
  --node-data dataex/clades_stability.parquet \
  --leaf-data dataex/leafdata.parquet \
  --output out/celltreeviz_example.html
```

## Data Conventions

For SciPy linkage input, leaf IDs are `0..n-1` and internal node IDs are
`n..2n-2`, matching `scipy.cluster.hierarchy` conventions. The node/clade table
should contain a `node_id` column by default.

The example leaf table is a wide matrix: rows are heatmap features, cell IDs are
columns, and `pathway` stores the feature label. `visualize()` also accepts
leaf-by-feature tables when `leaf_id_col` is provided or the dataframe index
contains leaf IDs.

By default, nodes and branches with no node-level metadata are hidden:

```python
visualize(..., show_missing_nodes=False)
```

Set `show_missing_nodes=True` to keep a muted full tree skeleton behind the
annotated clades.

## Continuous Colors

Continuous color specs are dictionaries keyed by column name. Values beyond
`vmin` and `vmax` are clipped by the Plotly color scale.

```python
visualize(
    "dataex/Z.npy",
    node_data="dataex/clades_stability.parquet",
    node_color="stability",
    continuous_colors={
        "stability": {
            "low": "#2166ac",
            "mid": "#f7f7f7",
            "high": "#b2182b",
            "mid_value": 0.0,
            "vmin": 0.6,
            "vmax": 1.0,
        }
    },
)
```

## Categorical Colors

Use `write_color_map()` and `read_color_map()` for YAML color dictionaries:

```python
from celltreeviz import write_color_map, visualize

write_color_map(
    "annotation_colors.yaml",
    {
        "annotation": {
            "lymphoid": "#4e79a7",
            "myeloid": "#f28e2b",
        }
    },
)

visualize(
    "dataex/Z.npy",
    node_data="dataex/clades_stability.parquet",
    categorical_colors="annotation_colors.yaml",
)
```

## Serving

For standalone use, write the HTML file. For a temporary local web server:

```python
visualize(..., output="out/app.html", serve=True, open_browser=True)
```

The server uses Python's standard library and blocks until interrupted.
