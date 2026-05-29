from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltreeviz import visualize


visualize(
    "dataex/Z.npy",
    node_data="dataex/clades_stability.parquet",
    leaf_data="dataex/leafdata.parquet",
    output="out/celltreeviz_example.html",
    node_size="size",
    node_color="annotation",
    edge_size="size",
    edge_color="annotation",
    show_missing_nodes=False,
)
