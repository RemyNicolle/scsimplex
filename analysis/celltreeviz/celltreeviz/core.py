from __future__ import annotations

import socket
import threading
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list


DEFAULT_CATEGORICAL_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
    "#2f4b7c",
    "#665191",
    "#a05195",
    "#d45087",
    "#f95d6a",
    "#ff7c43",
    "#ffa600",
]

DEFAULT_CONTINUOUS_SCALE = {
    "low": "#2166ac",
    "mid": "#f7f7f7",
    "high": "#b2182b",
    "mid_value": 0.0,
}


@dataclass(frozen=True)
class _TreeLayout:
    node_ids: list[int]
    x: dict[int, float]
    y: dict[int, float]
    parent: dict[int, int]
    children: dict[int, tuple[int, ...]]
    leaf_ids: list[int]
    leaf_order: list[int]
    leaf_counts: dict[int, int]
    root_id: int | None
    is_tree: bool


@dataclass(frozen=True)
class _ColorEncoding:
    kind: str
    values: pd.Series
    colors: dict[Any, str] | None
    colorscale: list[list[Any]] | None
    cmin: float | None
    cmax: float | None
    colorbar_title: str | None


def visualize(
    tree: str | Path | np.ndarray | nx.Graph,
    *,
    leaf_data: str | Path | pd.DataFrame | None = None,
    node_data: str | Path | pd.DataFrame | None = None,
    output: str | Path = "celltreeviz.html",
    node_id_col: str = "node_id",
    node_size: str | None = "size",
    node_color: str | None = "annotation",
    edge_size: str | None = None,
    edge_color: str | None = None,
    show_missing_nodes: bool = False,
    show_leaf_nodes: bool = False,
    leaf_feature_col: str | None = "pathway",
    leaf_id_col: str | None = None,
    leaf_labels: Sequence[str] | None = None,
    heatmap_features: Sequence[str] | None = None,
    heatmap_colors: Mapping[str, Any] | None = None,
    categorical_colors: Mapping[str, Mapping[Any, str]] | str | Path | None = None,
    continuous_colors: Mapping[str, Mapping[str, Any]] | None = None,
    clade_col: str | None = "clade",
    hover_columns: Sequence[str] | None = None,
    hover_max_items: int = 12,
    max_edge_color_bins: int = 12,
    max_edge_width_bins: int = 5,
    width: int = 1400,
    height: int = 900,
    title: str | None = None,
    include_plotlyjs: bool | str = True,
    serve: bool = False,
    port: int = 8050,
    open_browser: bool = False,
) -> Path | str:
    """Build an interactive web visualisation for a cell tree or graph.

    Parameters
    ----------
    tree:
        SciPy linkage matrix, path to a ``.npy`` linkage file, or a NetworkX
        graph. Linkage inputs receive a dendrogram layout with leaves aligned
        to the heatmap. Graph inputs receive a spring layout.
    leaf_data:
        Optional leaf-level data. Wide matrices are supported when rows are
        features and columns are leaf/cell IDs, with ``leaf_feature_col`` naming
        the feature label column. Leaf-by-feature tables are supported with
        ``leaf_id_col`` or leaf IDs in the index.
    node_data:
        Optional node/clade table. ``node_id_col`` is matched against linkage
        node IDs, where leaves are ``0..n-1`` and internal nodes are
        ``n..2n-2``.
    output:
        Destination HTML file.
    node_size, node_color, edge_size, edge_color:
        Column names in ``node_data``. Edges use child-node values by default.
        ``edge_size`` and ``edge_color`` fall back to the node settings.
    show_missing_nodes:
        When false, nodes and branches without a matching ``node_data`` row are
        hidden. When true, they are shown in a muted default style.
    categorical_colors:
        A nested mapping or YAML path such as
        ``{"annotation": {"myeloid": "#4e79a7"}}``.
    continuous_colors:
        Per-column linear color scale specs. Each spec can contain ``low``,
        ``high``, optional ``mid`` and ``mid_value``, plus ``vmin`` and ``vmax``.
    serve:
        If true, serve the output directory with Python's standard HTTP server
        and block until interrupted.

    Returns
    -------
    pathlib.Path | str
        The output path, or the local URL when ``serve=True``.
    """

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    linkage_or_graph = _load_tree(tree)
    node_df = _load_table(node_data)
    leaf_df = _load_table(leaf_data)
    categorical_maps = _load_color_maps(categorical_colors)
    continuous_specs = dict(continuous_colors or {})
    heatmap_specs = dict(heatmap_colors or {})

    if isinstance(linkage_or_graph, np.ndarray):
        layout = _layout_from_linkage(linkage_or_graph)
        inferred_leaf_labels = _infer_leaf_labels(
            leaf_df=leaf_df,
            n_leaves=len(layout.leaf_ids),
            leaf_feature_col=leaf_feature_col,
            leaf_id_col=leaf_id_col,
            explicit=leaf_labels,
        )
    else:
        layout = _layout_from_graph(linkage_or_graph)
        inferred_leaf_labels = _infer_leaf_labels(
            leaf_df=leaf_df,
            n_leaves=len(layout.leaf_ids),
            leaf_feature_col=leaf_feature_col,
            leaf_id_col=leaf_id_col,
            explicit=leaf_labels,
        )

    leaf_label_by_id = {
        leaf_id: inferred_leaf_labels[pos]
        for pos, leaf_id in enumerate(sorted(layout.leaf_ids))
        if pos < len(inferred_leaf_labels)
    }
    ordered_leaf_labels = [
        leaf_label_by_id.get(leaf_id, str(leaf_id)) for leaf_id in layout.leaf_order
    ]

    node_lookup = _prepare_node_data(node_df, node_id_col)
    draw_missing = show_missing_nodes or not node_lookup
    node_ids_to_draw = _nodes_to_draw(
        layout=layout,
        node_lookup=node_lookup,
        show_missing_nodes=draw_missing,
        show_leaf_nodes=show_leaf_nodes,
    )

    edge_size = node_size if edge_size is None else edge_size
    edge_color = node_color if edge_color is None else edge_color

    heatmap = _build_leaf_heatmap(
        leaf_df=leaf_df,
        ordered_leaf_labels=ordered_leaf_labels,
        leaf_feature_col=leaf_feature_col,
        leaf_id_col=leaf_id_col,
        heatmap_features=heatmap_features,
    )

    fig = _build_figure(
        layout=layout,
        node_lookup=node_lookup,
        node_ids_to_draw=node_ids_to_draw,
        leaf_label_by_id=leaf_label_by_id,
        ordered_leaf_labels=ordered_leaf_labels,
        node_size=node_size,
        node_color=node_color,
        edge_size=edge_size,
        edge_color=edge_color,
        categorical_maps=categorical_maps,
        continuous_specs=continuous_specs,
        heatmap=heatmap,
        heatmap_specs=heatmap_specs,
        clade_col=clade_col,
        hover_columns=hover_columns,
        hover_max_items=hover_max_items,
        max_edge_color_bins=max_edge_color_bins,
        max_edge_width_bins=max_edge_width_bins,
        show_missing_nodes=draw_missing,
        width=width,
        height=height,
        title=title,
    )
    fig.write_html(
        str(output_path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )

    if serve:
        url = _serve_blocking(output_path, port=port, open_browser=open_browser)
        return url
    if open_browser:
        webbrowser.open(output_path.as_uri())
    return output_path


def read_color_map(path: str | Path) -> dict[str, dict[Any, str]]:
    """Read a YAML categorical color dictionary."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Color-map YAML must contain a mapping.")
    return {
        str(column): {str(label): str(color) for label, color in values.items()}
        for column, values in loaded.items()
    }


def write_color_map(
    path: str | Path, color_map: Mapping[str, Mapping[Any, str]]
) -> Path:
    """Write a YAML categorical color dictionary."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(column): {str(label): str(color) for label, color in values.items()}
        for column, values in color_map.items()
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serialisable, handle, sort_keys=True)
    return output_path


def _load_tree(tree: str | Path | np.ndarray | nx.Graph) -> np.ndarray | nx.Graph:
    if isinstance(tree, np.ndarray):
        return tree
    if isinstance(tree, nx.Graph):
        return tree
    path = Path(tree).expanduser()
    if path.suffix.lower() == ".npy":
        return np.load(path)
    raise ValueError(
        "tree must be a SciPy linkage matrix, a .npy linkage path, or a NetworkX graph."
    )


def _load_table(value: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def _load_color_maps(
    value: Mapping[str, Mapping[Any, str]] | str | Path | None,
) -> dict[str, dict[Any, str]]:
    if value is None:
        return {}
    if isinstance(value, (str, Path)):
        return read_color_map(value)
    return {
        str(column): {label: str(color) for label, color in mapping.items()}
        for column, mapping in value.items()
    }


def _layout_from_linkage(Z: np.ndarray) -> _TreeLayout:
    if Z.ndim != 2 or Z.shape[1] != 4:
        raise ValueError("SciPy linkage arrays must have shape (n_leaves - 1, 4).")

    n = Z.shape[0] + 1
    leaf_order = [int(i) for i in leaves_list(Z)]
    max_height = float(np.nanmax(Z[:, 2])) if len(Z) else 1.0
    if not np.isfinite(max_height) or max_height <= 0:
        max_height = 1.0

    x: dict[int, float] = {leaf_id: max_height for leaf_id in range(n)}
    y: dict[int, float] = {
        leaf_id: float(position) for position, leaf_id in enumerate(leaf_order)
    }
    parent: dict[int, int] = {}
    children: dict[int, tuple[int, ...]] = {}
    leaf_counts: dict[int, int] = {leaf_id: 1 for leaf_id in range(n)}

    for row_idx, row in enumerate(Z):
        node_id = n + row_idx
        child_a, child_b = int(row[0]), int(row[1])
        height = float(row[2]) if np.isfinite(row[2]) else 0.0
        node_children = (child_a, child_b)
        children[node_id] = node_children
        parent[child_a] = node_id
        parent[child_b] = node_id
        leaf_counts[node_id] = int(row[3]) if np.isfinite(row[3]) else (
            leaf_counts[child_a] + leaf_counts[child_b]
        )
        x[node_id] = max_height - height
        y[node_id] = (y[child_a] + y[child_b]) / 2.0

    node_ids = list(range(2 * n - 1))
    root_id = 2 * n - 2 if n else None
    return _TreeLayout(
        node_ids=node_ids,
        x=x,
        y=y,
        parent=parent,
        children=children,
        leaf_ids=list(range(n)),
        leaf_order=leaf_order,
        leaf_counts=leaf_counts,
        root_id=root_id,
        is_tree=True,
    )


def _layout_from_graph(graph: nx.Graph) -> _TreeLayout:
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph input contains no nodes.")

    graph = nx.convert_node_labels_to_integers(graph, label_attribute="_label")
    positions = nx.spring_layout(graph, seed=7)
    node_ids = [int(node) for node in graph.nodes]

    if graph.is_directed():
        leaf_ids = [int(node) for node in graph.nodes if graph.out_degree(node) == 0]
    else:
        leaf_ids = [int(node) for node in graph.nodes if graph.degree(node) <= 1]
    if not leaf_ids:
        leaf_ids = node_ids
    raw_x = {int(node): float(pos[0]) for node, pos in positions.items()}
    raw_y = {int(node): float(pos[1]) for node, pos in positions.items()}
    leaf_order = sorted(leaf_ids, key=lambda node: raw_y[node])
    x = _rescale(raw_x, 0.0, 1.0)
    y = _rescale(raw_y, 0.0, float(max(len(leaf_order) - 1, 1)))
    for rank, leaf_id in enumerate(leaf_order):
        y[leaf_id] = float(rank)

    parent: dict[int, int] = {}
    children: dict[int, tuple[int, ...]] = {}
    for source in graph.nodes:
        successors = list(graph.successors(source)) if graph.is_directed() else list(graph.neighbors(source))
        children[int(source)] = tuple(int(target) for target in successors)
        for target in successors:
            parent.setdefault(int(target), int(source))

    return _TreeLayout(
        node_ids=node_ids,
        x=x,
        y=y,
        parent=parent,
        children=children,
        leaf_ids=leaf_ids,
        leaf_order=leaf_order,
        leaf_counts={node: 1 for node in node_ids},
        root_id=None,
        is_tree=False,
    )


def _infer_leaf_labels(
    *,
    leaf_df: pd.DataFrame | None,
    n_leaves: int,
    leaf_feature_col: str | None,
    leaf_id_col: str | None,
    explicit: Sequence[str] | None,
) -> list[str]:
    if explicit is not None:
        labels = [str(label) for label in explicit]
        if len(labels) != n_leaves:
            raise ValueError(
                f"leaf_labels has {len(labels)} labels, but the tree has {n_leaves} leaves."
            )
        return labels

    if leaf_df is not None:
        if leaf_feature_col and leaf_feature_col in leaf_df.columns:
            labels = [str(column) for column in leaf_df.columns if column != leaf_feature_col]
            if len(labels) >= n_leaves:
                return labels[:n_leaves]
        if leaf_id_col and leaf_id_col in leaf_df.columns:
            labels = leaf_df[leaf_id_col].astype(str).tolist()
            if len(labels) >= n_leaves:
                return labels[:n_leaves]
        if len(leaf_df.index) >= n_leaves:
            labels = [str(value) for value in leaf_df.index[:n_leaves]]
            if len(set(labels)) == len(labels):
                return labels

    return [str(i) for i in range(n_leaves)]


def _prepare_node_data(
    node_df: pd.DataFrame | None, node_id_col: str
) -> dict[int, pd.Series]:
    if node_df is None:
        return {}
    if node_id_col in node_df.columns:
        keyed = node_df.dropna(subset=[node_id_col]).copy()
        keyed[node_id_col] = keyed[node_id_col].astype(int)
        return {int(row[node_id_col]): row for _, row in keyed.iterrows()}
    if node_df.index.name == node_id_col or np.issubdtype(node_df.index.dtype, np.number):
        return {int(index): row for index, row in node_df.iterrows()}
    raise ValueError(
        f"node_data must contain a '{node_id_col}' column or use numeric node IDs as the index."
    )


def _nodes_to_draw(
    *,
    layout: _TreeLayout,
    node_lookup: Mapping[int, pd.Series],
    show_missing_nodes: bool,
    show_leaf_nodes: bool,
) -> list[int]:
    if show_missing_nodes:
        ids = list(layout.node_ids)
    else:
        ids = [node_id for node_id in layout.node_ids if node_id in node_lookup]
    if not show_leaf_nodes:
        leaf_set = set(layout.leaf_ids)
        ids = [node_id for node_id in ids if node_id not in leaf_set]
    return ids


def _rescale(values: Mapping[int, float], target_min: float, target_max: float) -> dict[int, float]:
    source = list(values.values())
    lo, hi = min(source), max(source)
    if lo == hi:
        midpoint = 0.5 * (target_min + target_max)
        return {key: midpoint for key in values}
    return {
        key: target_min + (value - lo) / (hi - lo) * (target_max - target_min)
        for key, value in values.items()
    }


def _build_leaf_heatmap(
    *,
    leaf_df: pd.DataFrame | None,
    ordered_leaf_labels: Sequence[str],
    leaf_feature_col: str | None,
    leaf_id_col: str | None,
    heatmap_features: Sequence[str] | None,
) -> pd.DataFrame | None:
    if leaf_df is None:
        return None

    labels = [str(label) for label in ordered_leaf_labels]
    if leaf_feature_col and leaf_feature_col in leaf_df.columns:
        table = leaf_df.copy()
        table[leaf_feature_col] = table[leaf_feature_col].astype(str)
        if heatmap_features is not None:
            wanted = [str(feature) for feature in heatmap_features]
            table = table[table[leaf_feature_col].isin(wanted)]
        feature_names = table[leaf_feature_col].tolist()
        value_columns = [label for label in labels if label in table.columns]
        if not value_columns:
            raise ValueError(
                "No heatmap columns match the tree leaf labels. Pass leaf_labels "
                "or check leaf_feature_col."
            )
        matrix = table.set_index(leaf_feature_col)[value_columns].T
        matrix = matrix.reindex(labels)
        matrix.columns = feature_names[: len(matrix.columns)]
        return matrix

    table = leaf_df.copy()
    if leaf_id_col and leaf_id_col in table.columns:
        table = table.set_index(leaf_id_col)
    table.index = table.index.astype(str)
    table = table.reindex(labels)
    if heatmap_features is not None:
        table = table[[feature for feature in heatmap_features if feature in table.columns]]
    return table


def _build_figure(
    *,
    layout: _TreeLayout,
    node_lookup: Mapping[int, pd.Series],
    node_ids_to_draw: Sequence[int],
    leaf_label_by_id: Mapping[int, str],
    ordered_leaf_labels: Sequence[str],
    node_size: str | None,
    node_color: str | None,
    edge_size: str | None,
    edge_color: str | None,
    categorical_maps: Mapping[str, Mapping[Any, str]],
    continuous_specs: Mapping[str, Mapping[str, Any]],
    heatmap: pd.DataFrame | None,
    heatmap_specs: Mapping[str, Any],
    clade_col: str | None,
    hover_columns: Sequence[str] | None,
    hover_max_items: int,
    max_edge_color_bins: int,
    max_edge_width_bins: int,
    show_missing_nodes: bool,
    width: int,
    height: int,
    title: str | None,
) -> go.Figure:
    has_heatmap = heatmap is not None and not heatmap.empty
    fig = make_subplots(
        rows=1,
        cols=2 if has_heatmap else 1,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        column_widths=[0.42, 0.58] if has_heatmap else [1.0],
    )

    edge_traces = _edge_traces(
        layout=layout,
        node_lookup=node_lookup,
        edge_size=edge_size,
        edge_color=edge_color,
        categorical_maps=categorical_maps,
        continuous_specs=continuous_specs,
        show_missing_nodes=show_missing_nodes,
        max_edge_color_bins=max_edge_color_bins,
        max_edge_width_bins=max_edge_width_bins,
    )
    for trace in edge_traces:
        fig.add_trace(trace, row=1, col=1)

    node_trace, legend_traces = _node_traces(
        layout=layout,
        node_lookup=node_lookup,
        node_ids_to_draw=node_ids_to_draw,
        leaf_label_by_id=leaf_label_by_id,
        node_size=node_size,
        node_color=node_color,
        categorical_maps=categorical_maps,
        continuous_specs=continuous_specs,
        clade_col=clade_col,
        hover_columns=hover_columns,
        hover_max_items=hover_max_items,
    )
    if node_trace is not None:
        fig.add_trace(node_trace, row=1, col=1)
    for trace in legend_traces:
        fig.add_trace(trace, row=1, col=1)

    if has_heatmap:
        for trace in _heatmap_traces(
            heatmap=heatmap,
            heatmap_specs=heatmap_specs,
            categorical_maps=categorical_maps,
        ):
            fig.add_trace(trace, row=1, col=2)

    y_range = [-1, max(len(ordered_leaf_labels) - 0.5, 1)]
    fig.update_yaxes(
        range=y_range,
        showticklabels=False,
        autorange=False,
        title_text="Leaves",
        row=1,
        col=1,
    )
    if has_heatmap:
        fig.update_yaxes(
            range=y_range,
            showticklabels=False,
            autorange=False,
            row=1,
            col=2,
        )

    fig.update_xaxes(
        title_text="Tree distance" if layout.is_tree else "Graph x",
        zeroline=False,
        showgrid=False,
        row=1,
        col=1,
    )
    if has_heatmap:
        fig.update_xaxes(
            title_text="Leaf-level features",
            tickangle=45,
            side="top",
            row=1,
            col=2,
        )

    fig.update_layout(
        title=title or "Cell hierarchy visualisation",
        template="plotly_white",
        width=width,
        height=height,
        hovermode="closest",
        dragmode="pan",
        legend_title_text="Categories",
        margin={"l": 50, "r": 30, "t": 80, "b": 50},
    )
    return fig


def _edge_traces(
    *,
    layout: _TreeLayout,
    node_lookup: Mapping[int, pd.Series],
    edge_size: str | None,
    edge_color: str | None,
    categorical_maps: Mapping[str, Mapping[Any, str]],
    continuous_specs: Mapping[str, Mapping[str, Any]],
    show_missing_nodes: bool,
    max_edge_color_bins: int,
    max_edge_width_bins: int,
) -> list[go.Scattergl]:
    edge_rows: list[dict[str, Any]] = []
    for child_id, parent_id in layout.parent.items():
        if not show_missing_nodes and child_id not in node_lookup:
            continue
        row = node_lookup.get(child_id)
        color_value = _series_get(row, edge_color)
        size_value = _series_get(row, edge_size)
        edge_rows.append(
            {
                "child_id": child_id,
                "parent_id": parent_id,
                "color_value": color_value,
                "size_value": size_value,
                "has_data": row is not None,
            }
        )

    if not edge_rows:
        return []

    color_values = pd.Series([row["color_value"] for row in edge_rows])
    size_values = pd.Series([row["size_value"] for row in edge_rows])
    color_encoding = _make_color_encoding(
        edge_color or "",
        color_values,
        categorical_maps=categorical_maps,
        continuous_specs=continuous_specs,
    )
    widths = _scaled_widths(size_values, max_bins=max_edge_width_bins)

    grouped: dict[tuple[str, float], dict[str, list[Any]]] = {}
    for idx, edge in enumerate(edge_rows):
        color = _edge_color_for_value(
            value=edge["color_value"],
            encoding=color_encoding,
            bin_count=max_edge_color_bins,
            has_data=edge["has_data"],
        )
        width = widths[idx] if edge["has_data"] else 0.6
        key = (color, width)
        group = grouped.setdefault(key, {"x": [], "y": []})
        child_id = edge["child_id"]
        parent_id = edge["parent_id"]
        child_x, child_y = layout.x[child_id], layout.y[child_id]
        parent_x, parent_y = layout.x[parent_id], layout.y[parent_id]
        if layout.is_tree:
            group["x"].extend([child_x, parent_x, None, parent_x, parent_x, None])
            group["y"].extend([child_y, child_y, None, child_y, parent_y, None])
        else:
            group["x"].extend([child_x, parent_x, None])
            group["y"].extend([child_y, parent_y, None])

    traces: list[go.Scattergl] = []
    for (color, width), coords in grouped.items():
        traces.append(
            go.Scattergl(
                x=coords["x"],
                y=coords["y"],
                mode="lines",
                line={"color": color, "width": width},
                hoverinfo="skip",
                showlegend=False,
                name="branch",
            )
        )
    return traces


def _node_traces(
    *,
    layout: _TreeLayout,
    node_lookup: Mapping[int, pd.Series],
    node_ids_to_draw: Sequence[int],
    leaf_label_by_id: Mapping[int, str],
    node_size: str | None,
    node_color: str | None,
    categorical_maps: Mapping[str, Mapping[Any, str]],
    continuous_specs: Mapping[str, Mapping[str, Any]],
    clade_col: str | None,
    hover_columns: Sequence[str] | None,
    hover_max_items: int,
) -> tuple[go.Scattergl | None, list[go.Scattergl]]:
    if not node_ids_to_draw:
        return None, []

    rows = [node_lookup.get(node_id) for node_id in node_ids_to_draw]
    color_values = pd.Series([_series_get(row, node_color) for row in rows])
    size_values = pd.Series([_series_get(row, node_size) for row in rows])
    encoding = _make_color_encoding(
        node_color or "",
        color_values,
        categorical_maps=categorical_maps,
        continuous_specs=continuous_specs,
    )
    marker_sizes = _scaled_marker_sizes(size_values)
    hover_text = [
        _node_hover(
            node_id=node_id,
            row=row,
            leaf_label=leaf_label_by_id.get(node_id),
            leaf_count=layout.leaf_counts.get(node_id),
            clade_col=clade_col,
            hover_columns=hover_columns,
            hover_max_items=hover_max_items,
        )
        for node_id, row in zip(node_ids_to_draw, rows)
    ]

    marker: dict[str, Any] = {
        "size": marker_sizes,
        "line": {"width": 0.8, "color": "#1f2933"},
        "opacity": 0.9,
    }
    legend_traces: list[go.Scattergl] = []
    if encoding.kind == "continuous":
        numeric = pd.to_numeric(color_values, errors="coerce")
        marker.update(
            {
                "color": numeric,
                "colorscale": encoding.colorscale,
                "cmin": encoding.cmin,
                "cmax": encoding.cmax,
                "colorbar": {"title": encoding.colorbar_title or node_color},
            }
        )
    elif encoding.kind == "categorical":
        color_map = encoding.colors or {}
        marker["color"] = [
            color_map.get(value, "#aab2bd") if pd.notna(value) else "#d5d9de"
            for value in color_values
        ]
        legend_traces = _legend_marker_traces(node_color or "category", color_map)
    else:
        marker["color"] = ["#2b6cb0" if row is not None else "#d5d9de" for row in rows]

    return (
        go.Scattergl(
            x=[layout.x[node_id] for node_id in node_ids_to_draw],
            y=[layout.y[node_id] for node_id in node_ids_to_draw],
            mode="markers",
            marker=marker,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="nodes",
            showlegend=False,
        ),
        legend_traces,
    )


def _heatmap_traces(
    *,
    heatmap: pd.DataFrame,
    heatmap_specs: Mapping[str, Any],
    categorical_maps: Mapping[str, Mapping[Any, str]],
) -> list[go.Heatmap]:
    y_positions = list(range(len(heatmap.index)))
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in heatmap.dtypes):
        numeric = heatmap.apply(pd.to_numeric, errors="coerce")
        scale = _continuous_scale_from_spec("heatmap", numeric.stack(), heatmap_specs)
        return [
            go.Heatmap(
                z=numeric.to_numpy(),
                x=[str(column) for column in numeric.columns],
                y=y_positions,
                colorscale=scale.colorscale,
                zmin=scale.cmin,
                zmax=scale.cmax,
                colorbar={"title": "Leaf data"},
                hovertemplate="leaf row %{y}<br>%{x}: %{z}<extra></extra>",
                name="leaf heatmap",
                showscale=True,
            )
        ]

    traces: list[go.Heatmap] = []
    for col_idx, column in enumerate(heatmap.columns):
        values = heatmap[column]
        x_label = str(column)
        if pd.api.types.is_numeric_dtype(values):
            scale = _continuous_scale_from_spec(x_label, values, heatmap_specs)
            traces.append(
                go.Heatmap(
                    z=pd.to_numeric(values, errors="coerce").to_numpy()[:, np.newaxis],
                    x=[x_label],
                    y=y_positions,
                    colorscale=scale.colorscale,
                    zmin=scale.cmin,
                    zmax=scale.cmax,
                    showscale=col_idx == 0,
                    hovertemplate="leaf row %{y}<br>%{x}: %{z}<extra></extra>",
                    name=x_label,
                )
            )
            continue

        color_map = _categorical_map_for_column(x_label, values, categorical_maps)
        encoded = values.map({label: idx for idx, label in enumerate(color_map)})
        colorscale = _categorical_colorscale(list(color_map.values()))
        traces.append(
            go.Heatmap(
                z=encoded.to_numpy()[:, np.newaxis],
                x=[x_label],
                y=y_positions,
                colorscale=colorscale,
                zmin=-0.5,
                zmax=max(len(color_map) - 0.5, 0.5),
                showscale=False,
                hovertemplate="leaf row %{y}<br>%{x}<extra></extra>",
                name=x_label,
            )
        )
    return traces


def _make_color_encoding(
    column: str,
    values: pd.Series,
    *,
    categorical_maps: Mapping[str, Mapping[Any, str]],
    continuous_specs: Mapping[str, Mapping[str, Any]],
) -> _ColorEncoding:
    non_missing = values.dropna()
    if not column or non_missing.empty:
        return _ColorEncoding("none", values, None, None, None, None, None)

    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.notna().all() and column not in categorical_maps:
        scale = _continuous_scale_from_spec(column, numeric, continuous_specs.get(column, {}))
        return _ColorEncoding(
            "continuous",
            values,
            None,
            scale.colorscale,
            scale.cmin,
            scale.cmax,
            column,
        )

    return _ColorEncoding(
        "categorical",
        values,
        _categorical_map_for_column(column, non_missing, categorical_maps),
        None,
        None,
        None,
        column,
    )


def _continuous_scale_from_spec(
    column: str, values: pd.Series, specs: Mapping[str, Any]
) -> _ColorEncoding:
    if column in specs and isinstance(specs[column], Mapping):
        spec = {**DEFAULT_CONTINUOUS_SCALE, **dict(specs[column])}
    else:
        spec = {**DEFAULT_CONTINUOUS_SCALE, **dict(specs or {})}

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        cmin, cmax = 0.0, 1.0
    else:
        cmin = float(spec.get("vmin", numeric.min()))
        cmax = float(spec.get("vmax", numeric.max()))
    if cmin == cmax:
        cmin -= 0.5
        cmax += 0.5

    mid = spec.get("mid")
    mid_value = spec.get("mid_value", 0.0)
    if mid is not None and cmin < float(mid_value) < cmax:
        midpoint = (float(mid_value) - cmin) / (cmax - cmin)
        colorscale = [
            [0.0, spec["low"]],
            [midpoint, mid],
            [1.0, spec["high"]],
        ]
    elif "mid" in spec and cmin < float(mid_value) < cmax:
        midpoint = (float(mid_value) - cmin) / (cmax - cmin)
        colorscale = [
            [0.0, spec["low"]],
            [midpoint, spec["mid"]],
            [1.0, spec["high"]],
        ]
    else:
        colorscale = [[0.0, spec["low"]], [1.0, spec["high"]]]

    return _ColorEncoding("continuous", values, None, colorscale, cmin, cmax, column)


def _categorical_map_for_column(
    column: str,
    values: Iterable[Any],
    categorical_maps: Mapping[str, Mapping[Any, str]],
) -> dict[Any, str]:
    existing = dict(categorical_maps.get(column, {}))
    labels = [value for value in pd.Series(list(values)).dropna().unique()]
    color_map: dict[Any, str] = {}
    for idx, label in enumerate(labels):
        color_map[label] = existing.get(label) or existing.get(str(label)) or DEFAULT_CATEGORICAL_COLORS[
            idx % len(DEFAULT_CATEGORICAL_COLORS)
        ]
    for label, color in existing.items():
        color_map.setdefault(label, color)
    return color_map


def _categorical_colorscale(colors: Sequence[str]) -> list[list[Any]]:
    if not colors:
        return [[0.0, "#d5d9de"], [1.0, "#d5d9de"]]
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]
    scale: list[list[Any]] = []
    n = len(colors)
    for idx, color in enumerate(colors):
        left = idx / n
        right = (idx + 1) / n
        scale.append([left, color])
        scale.append([right, color])
    scale[-1][0] = 1.0
    return scale


def _edge_color_for_value(
    *,
    value: Any,
    encoding: _ColorEncoding,
    bin_count: int,
    has_data: bool,
) -> str:
    if not has_data:
        return "#d5d9de"
    if encoding.kind == "categorical":
        return (encoding.colors or {}).get(value, "#aab2bd")
    if encoding.kind == "continuous":
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "#aab2bd"
        cmin = encoding.cmin if encoding.cmin is not None else float(numeric)
        cmax = encoding.cmax if encoding.cmax is not None else float(numeric)
        if cmax == cmin:
            t = 0.5
        else:
            t = max(0.0, min(1.0, (float(numeric) - cmin) / (cmax - cmin)))
        if bin_count > 1:
            t = round(t * (bin_count - 1)) / (bin_count - 1)
        return _sample_colorscale(encoding.colorscale or [[0, "#2166ac"], [1, "#b2182b"]], t)
    return "#6b7785"


def _sample_colorscale(colorscale: Sequence[Sequence[Any]], t: float) -> str:
    stops = sorted((float(stop), str(color)) for stop, color in colorscale)
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for (left_t, left_color), (right_t, right_color) in zip(stops, stops[1:]):
        if left_t <= t <= right_t:
            local = (t - left_t) / (right_t - left_t) if right_t > left_t else 0.0
            return _interpolate_hex(left_color, right_color, local)
    return stops[-1][1]


def _interpolate_hex(left: str, right: str, t: float) -> str:
    left_rgb = _hex_to_rgb(left)
    right_rgb = _hex_to_rgb(right)
    rgb = [
        int(round(left_rgb[idx] + (right_rgb[idx] - left_rgb[idx]) * t))
        for idx in range(3)
    ]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip()
    if not value.startswith("#"):
        return (107, 119, 133)
    value = value[1:]
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) != 6:
        return (107, 119, 133)
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _scaled_marker_sizes(values: pd.Series, minimum: float = 5, maximum: float = 22) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return [9.0 if pd.notna(value) else 6.0 for value in values]
    lo, hi = float(numeric.min()), float(numeric.max())
    if lo == hi:
        return [0.5 * (minimum + maximum) if pd.notna(value) else 6.0 for value in numeric]
    return [
        minimum + (float(value) - lo) / (hi - lo) * (maximum - minimum)
        if pd.notna(value)
        else 6.0
        for value in numeric
    ]


def _scaled_widths(values: pd.Series, max_bins: int, minimum: float = 0.7, maximum: float = 4.5) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return [1.2 for _ in values]
    lo, hi = float(numeric.min()), float(numeric.max())
    if lo == hi:
        return [0.5 * (minimum + maximum) if pd.notna(value) else 0.7 for value in numeric]
    widths = []
    for value in numeric:
        if pd.isna(value):
            widths.append(0.7)
            continue
        t = (float(value) - lo) / (hi - lo)
        if max_bins > 1:
            t = round(t * (max_bins - 1)) / (max_bins - 1)
        widths.append(minimum + t * (maximum - minimum))
    return widths


def _legend_marker_traces(column: str, color_map: Mapping[Any, str]) -> list[go.Scattergl]:
    return [
        go.Scattergl(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 8, "color": color},
            name=f"{column}: {label}",
            showlegend=True,
            hoverinfo="skip",
        )
        for label, color in color_map.items()
    ]


def _node_hover(
    *,
    node_id: int,
    row: pd.Series | None,
    leaf_label: str | None,
    leaf_count: int | None,
    clade_col: str | None,
    hover_columns: Sequence[str] | None,
    hover_max_items: int,
) -> str:
    lines = [f"<b>node {node_id}</b>"]
    if leaf_label is not None:
        lines.append(f"leaf: {leaf_label}")
    if leaf_count is not None:
        lines.append(f"leaf count: {leaf_count}")
    if row is None:
        lines.append("no node data")
        return "<br>".join(lines)

    if hover_columns is None:
        columns = [
            column
            for column in row.index
            if column != clade_col and not isinstance(row[column], (list, tuple, set, np.ndarray))
        ]
    else:
        columns = [column for column in hover_columns if column in row.index]
    for column in columns:
        value = row[column]
        if _is_missing_scalar(value):
            continue
        lines.append(f"{column}: {_format_hover_value(value)}")

    if clade_col and clade_col in row.index:
        clade = row[clade_col]
        if isinstance(clade, np.ndarray):
            clade = clade.tolist()
        if isinstance(clade, (list, tuple, set)):
            items = list(clade)
            preview = ", ".join(str(item) for item in items[:hover_max_items])
            suffix = "..." if len(items) > hover_max_items else ""
            lines.append(f"{clade_col} ({len(items)}): {preview}{suffix}")
    return "<br>".join(lines)


def _format_hover_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _is_missing_scalar(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, np.ndarray, dict)):
        return False
    return bool(pd.isna(value))


def _series_get(row: pd.Series | None, column: str | None) -> Any:
    if row is None or not column or column not in row.index:
        return np.nan
    return row[column]


def _serve_blocking(output_path: Path, *, port: int, open_browser: bool) -> str:
    host = "127.0.0.1"
    chosen_port = _available_port(port)
    url = f"http://{host}:{chosen_port}/{output_path.name}"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(output_path.parent), **kwargs)

    server = ThreadingHTTPServer((host, chosen_port), Handler)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    print(f"Serving celltreeviz at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return url


def _available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
