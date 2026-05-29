from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from celltreeviz import read_color_map, visualize, write_color_map


def test_visualize_writes_html_for_linkage(tmp_path: Path) -> None:
    z = np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.2, 2],
            [4, 5, 0.5, 4],
        ],
        dtype=float,
    )
    leaf_data = pd.DataFrame(
        {
            "pathway": ["score_a", "score_b"],
            "cell_a": [0.1, 1.0],
            "cell_b": [0.2, 0.8],
            "cell_c": [0.8, 0.1],
            "cell_d": [1.0, 0.0],
        }
    )
    node_data = pd.DataFrame(
        {
            "node_id": [4, 6],
            "size": [2, 4],
            "stability": [0.7, 0.9],
            "annotation": ["left", "root"],
            "clade": [["cell_a", "cell_b"], ["cell_a", "cell_b", "cell_c", "cell_d"]],
        }
    )

    output = visualize(
        z,
        leaf_data=leaf_data,
        node_data=node_data,
        output=tmp_path / "toy.html",
        node_size="size",
        node_color="annotation",
        edge_size="size",
        edge_color="stability",
        leaf_labels=["cell_a", "cell_b", "cell_c", "cell_d"],
        include_plotlyjs=False,
    )

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Cell hierarchy visualisation" in html
    assert "score_a" in html
    assert "node 4" in html


def test_color_map_roundtrip(tmp_path: Path) -> None:
    path = write_color_map(
        tmp_path / "colors.yaml",
        {"annotation": {"left": "#4e79a7", "root": "#f28e2b"}},
    )

    assert read_color_map(path) == {
        "annotation": {"left": "#4e79a7", "root": "#f28e2b"}
    }


def test_visualize_draws_without_node_data(tmp_path: Path) -> None:
    z = np.array([[0, 1, 0.1, 2]], dtype=float)

    output = visualize(z, output=tmp_path / "no_node_data.html", include_plotlyjs=False)

    assert output.exists()
    assert "Tree distance" in output.read_text(encoding="utf-8")


def test_visualize_accepts_networkx_graph(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (2, 3)])

    output = visualize(graph, output=tmp_path / "graph.html", include_plotlyjs=False)

    assert output.exists()
    assert "Graph x" in output.read_text(encoding="utf-8")
