# Degree distributions and log-log CCDF figures used in the analysis notebooks.

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def degree_distribution(graph: nx.Graph, *, include_zero_degree: bool = True) -> pd.DataFrame:
    degrees = np.fromiter((deg for _, deg in graph.degree()), dtype=int)
    if not include_zero_degree:
        degrees = degrees[degrees > 0]
    if degrees.size == 0:
        return pd.DataFrame(columns=["degree", "count", "pdf", "ccdf"])

    unique, counts = np.unique(degrees, return_counts=True)
    pdf = counts / counts.sum()
    ccdf = np.array([np.mean(degrees >= degree) for degree in unique])
    return pd.DataFrame({"degree": unique, "count": counts, "pdf": pdf, "ccdf": ccdf})


def plot_degree_ccdf(distribution_df: pd.DataFrame, *, title: str, output_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    positive = distribution_df[(distribution_df["degree"] > 0) & (distribution_df["ccdf"] > 0)]
    if not positive.empty:
        ax.loglog(positive["degree"], positive["ccdf"], marker="o", linestyle="none")
    ax.set_xlabel("Degree")
    ax.set_ylabel("P(K ≥ k)")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax
