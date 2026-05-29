from __future__ import annotations

import argparse

from .core import visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive celltreeviz HTML visualisation."
    )
    parser.add_argument("tree", help="Path to a SciPy linkage .npy file.")
    parser.add_argument("--leaf-data", help="Leaf/cell-level parquet, CSV, or TSV.")
    parser.add_argument("--node-data", help="Node/clade-level parquet, CSV, or TSV.")
    parser.add_argument("--output", default="celltreeviz.html", help="Output HTML path.")
    parser.add_argument("--node-id-col", default="node_id")
    parser.add_argument("--node-size", default="size")
    parser.add_argument("--node-color", default="annotation")
    parser.add_argument("--edge-size")
    parser.add_argument("--edge-color")
    parser.add_argument("--leaf-feature-col", default="pathway")
    parser.add_argument("--leaf-id-col")
    parser.add_argument("--categorical-colors")
    parser.add_argument("--show-missing-nodes", action="store_true")
    parser.add_argument("--show-leaf-nodes", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    result = visualize(
        args.tree,
        leaf_data=args.leaf_data,
        node_data=args.node_data,
        output=args.output,
        node_id_col=args.node_id_col,
        node_size=args.node_size,
        node_color=args.node_color,
        edge_size=args.edge_size,
        edge_color=args.edge_color,
        leaf_feature_col=args.leaf_feature_col,
        leaf_id_col=args.leaf_id_col,
        categorical_colors=args.categorical_colors,
        show_missing_nodes=args.show_missing_nodes,
        show_leaf_nodes=args.show_leaf_nodes,
        serve=args.serve,
        open_browser=args.open_browser,
        port=args.port,
    )
    if not args.serve:
        print(result)


if __name__ == "__main__":
    main()
