# import relevant packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

## for the DAGs

import networkx as nx
# import os

# Part 1: Configurations of the DAGs and Underlying Functions

plt.rcParams["figure.dpi"] = 150

def build_dag(edges):
    '''
    Build a directed graph (networkx.DiGraph) from a list of (source, target) edges.
    Only nodes present in edges are added.
    '''
    G = nx.DiGraph()
    for u, v in edges:
        G.add_edge(u, v)
    return G


def draw_dag_ax(G, coords, ax, title=None, subtitle=None, panel_label=None):
    '''
    Draw a directed graph on a Matplotlib axis using fixed coordinates.

    - G: networkx.DiGraph
    - coords: dict with 'x' and 'y' dicts mapping node names to coordinates
    - ax: Matplotlib axis object
    - title: optional axis title (unused here)
    - subtitle: text shown under the DAG
    - panel_label: e.g. '1(a)', shown at top-left
    '''
    

    # Positions only for nodes that have coordinates
    pos = {
        n: (coords["x"][n], coords["y"][n])
        for n in G.nodes
        if n in coords["x"] and n in coords["y"]
    }

    # Draw edges where both endpoints have coordinates
    edgelist = [(u, v) for u, v in G.edges if u in pos and v in pos]
    if edgelist:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=edgelist,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            width=1.2,
        )

    # Draw nodes
    nodelist = list(pos.keys())
    if nodelist:
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=nodelist,
            ax=ax,
            node_size=400,
            node_color="white",
            edgecolors="black",
            linewidths=1.2,
        )
        nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=10)

    # Axes styling (no ticks, no spines)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Panel label (e.g., "1(a)")
    if panel_label:
        ax.text(-1.0, 2.0, panel_label, fontsize=12)

    # Subtitle under the DAG
    if subtitle:
        ax.text(0.0, -1.0, subtitle, fontsize=9, ha="center", va="center")

    # Small t₁..t₄ markers along the bottom
    ax.text(-1, -0.5, "t₁", fontsize=9)
    ax.text(0, -0.5, "t₂", fontsize=9)
    ax.text(1, -0.5, "t₃", fontsize=9)
    ax.text(2, -0.5, "t₄", fontsize=9)

    # Coordinate limits to match R layout
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.5)

    if title:
        ax.set_title(title, fontsize=12, pad=6)



# Figure 1
coords1 = dict(
    x=dict(Y=2, T=0, U=-1, F=1, X=0, T1=-1, T2=0),
    y=dict(Y=1, T=1, U=1, F=2, X=0, T1=-1, T2=-1),
)

# 1(a): Y ~ T + F + X; F ~ U; T ~ U + X
edges_1a = [
    ("T", "Y"), ("F", "Y"), ("X", "Y"),
    ("U", "F"),
    ("U", "T"), ("X", "T"),
]
G1a = build_dag(edges_1a)

# 1(b): Y ~ T + F + X; F ~ T; T ~ U + X
edges_1b = [
    ("T", "Y"), ("F", "Y"), ("X", "Y"),
    ("T", "F"),
    ("U", "T"), ("X", "T"),
]
G1b = build_dag(edges_1b)

# 1(c): Y ~ T + F + X; F ~ T + U; T ~ U + X
edges_1c = [
    ("T", "Y"), ("F", "Y"), ("X", "Y"),
    ("T", "F"), ("U", "F"),
    ("U", "T"), ("X", "T"),
]
G1c = build_dag(edges_1c)

# Build Figure 1 via subplots (same style as Figure 2)
fig1, axes = plt.subplots(2, 2, figsize=(7, 7.5))
ax = axes.ravel()

# Top-left: 1(a)
draw_dag_ax(
    G1a,
    coords1,
    ax[0],
    panel_label="1(a)",
    subtitle="Controlling for F and X identifies the effect of T",
)

# Top-right: 1(b)
draw_dag_ax(
    G1b,
    coords1,
    ax[1],
    panel_label="1(b)",
    subtitle="Controlling for X identifies the effect of T,\n"
             "controlling for F generates post-treatment bias",
)

# Bottom-left: 1(c)
draw_dag_ax(
    G1c,
    coords1,
    ax[2],
    panel_label="1(c)",
    subtitle="The effect of T is not identifiable\nusing observed covariates",
)

# Bottom-right: empty
ax[3].axis("off")

# Shared caption at the bottom
fig1_text = (
    "HPT seek to estimate the effect of Distance to camps (T) on Intolerance (Y).\n"
    "The issue is whether Länder fixed effects (F) create post-treatment bias, or whether they help to capture\n"
    "unobserved state-level factors (U) that also explain intolerance."
)
fig1.text(0.1, 0.06, fig1_text, ha="left", va="center", fontsize=9)

fig1.tight_layout(rect=[0, 0.1, 1, 1])
fig1.savefig("figure1.pdf")
print("Figure 1 saved to figure1.pdf")



# Coordinates for 2(a)
coords2a = dict(
    x=dict(Y=2, T=0, U1=-1, U2=0, F=1, X=0, T1=-1, T2=0),
    y=dict(Y=1, T=1, U1=1.5, U2=2, F=1.5, X=0, T1=-1, T2=-1),
)

# Coordinates for 2(b)–2(d)
coords2bcd = dict(
    x=dict(Y=2, T=0, U1=-1, U2=0, F=1, X=0, T1=-1, T2=0),
    y=dict(Y=1, T=1, U1=1.5, U2=2, F=1.25, X=0, T1=-1, T2=-1),
)

# 2(a): Y ~ T + F + X; F ~ U2; T ~ U1 + U2 + X
edges_2a = [
    ("T", "Y"), ("F", "Y"), ("X", "Y"),
    ("U2", "F"),
    ("U1", "T"), ("U2", "T"), ("X", "T"),
]
G2a = build_dag(edges_2a)

# 2(b): Y ~ T + X + U2; F ~ U1 + U2; T ~ U1 + X
edges_2b = [
    ("T", "Y"), ("X", "Y"), ("U2", "Y"),
    ("U1", "F"), ("U2", "F"),
    ("U1", "T"), ("X", "T"),
]
G2b = build_dag(edges_2b)

# 2(c): Y ~ T + F + X + U2; F ~ U2 + U1; T ~ U1 + X
edges_2c = [
    ("T", "Y"), ("F", "Y"), ("X", "Y"), ("U2", "Y"),
    ("U2", "F"), ("U1", "F"),
    ("U1", "T"), ("X", "T"),
]
G2c = build_dag(edges_2c)

# 2(d): Y ~ T + X + U2; F ~ U2 + U1; T ~ U1 + X + U2
edges_2d = [
    ("T", "Y"), ("X", "Y"), ("U2", "Y"),
    ("U2", "F"), ("U1", "F"),
    ("U1", "T"), ("X", "T"), ("U2", "T"),
]
G2d = build_dag(edges_2d)

# Build Figure 2 via 2x2 subplots
fig2, axes = plt.subplots(2, 2, figsize=(7, 7.5))
ax = axes.ravel()

draw_dag_ax(
    G2a,
    coords2a,
    ax[0],
    panel_label="2(a)",
    subtitle="Controlling for F, X identifies the effect of T",
)
draw_dag_ax(
    G2b,
    coords2bcd,
    ax[1],
    panel_label="2(b)",
    subtitle="Controlling for X identifies the effect of T,\n"
             "controlling for F generates M-bias",
)
draw_dag_ax(
    G2c,
    coords2bcd,
    ax[2],
    panel_label="2(c)",
    subtitle="The effect of T is not identifiable\nusing observed covariates",
)
draw_dag_ax(
    G2d,
    coords2bcd,
    ax[3],
    panel_label="2(d)",
    subtitle="The effect of T is not identifiable\nusing observed covariates",
)

fig2_text = (
    "HPT seek to estimate the effect of Distance to camps (T) on Intolerance (Y).\n"
    "The issue is whether Länder fixed effects (F) create M-bias, or whether they help to capture\n"
    "unobserved state-level factors (U) that also explain intolerance."
)
fig2.text(0.1, 0.06, fig2_text, ha="left", va="center", fontsize=9)

fig2.tight_layout(rect=[0, 0.1, 1, 1])
fig2.savefig("figure2.pdf")
print("Figure 2 saved to figure2.pdf")