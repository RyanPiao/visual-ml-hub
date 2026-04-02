"""
Interactive 3D Decision Tree Splits — Plotly

3D scatter plot (Income x Credit Score x Loan Amount) with translucent
split planes. Slider controls tree depth (1-10): as depth increases,
more planes appear, partitioning the space into smaller rectangular regions.

Students can: rotate, zoom, hover on points, drag the depth slider.

Usage:
    # Jupyter / Colab
    python tree_splits_3d.py           # opens in browser
    # or import and call:
    from tree_splits_3d import create_tree_splits_figure
    fig = create_tree_splits_figure(max_depth=5)
    fig.show()

    # Export for reveal.js slides
    fig.write_html("tree_splits.html", include_plotlyjs="cdn")

Course:  Econ 5200 / 3916
Topic:   Lecture 19 — Decision Trees: Recursive Partitioning
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout, make_split_plane_mesh

np.random.seed(42)


def generate_loan_data(n=200):
    """Synthetic loan default data with 3 features."""
    income = np.random.uniform(25, 85, n)
    credit = np.random.uniform(520, 780, n)
    loan = np.random.uniform(5, 35, n)

    score = (
        0.06 * (income - 52)
        + 0.008 * (credit - 640)
        - 0.05 * (loan - 20)
    )
    prob = 1 / (1 + np.exp(score))
    labels = (np.random.random(n) < prob).astype(int)

    X = np.column_stack([income, credit, loan])
    return X, labels


def extract_splits_from_sklearn(tree_model, feature_names, X):
    """Extract split planes from a fitted sklearn DecisionTreeClassifier."""
    tree = tree_model.tree_
    splits = []

    def _walk(node_id, bounds):
        if tree.feature[node_id] < 0:  # leaf
            return

        feat = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        depth_val = _get_depth(node_id)

        splits.append({
            "axis": feat,
            "axis_name": feature_names[feat],
            "value": thresh,
            "bounds": dict(bounds),
            "depth": depth_val,
            "samples": tree.n_node_samples[node_id],
        })

        # Left child: feat <= thresh
        left_bounds = dict(bounds)
        keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        left_bounds[keys[feat * 2 + 1]] = thresh

        right_bounds = dict(bounds)
        right_bounds[keys[feat * 2]] = thresh

        _walk(tree.children_left[node_id], left_bounds)
        _walk(tree.children_right[node_id], right_bounds)

    def _get_depth(node_id):
        """Compute depth of a node by walking up the tree."""
        d = 0
        parents = np.full(tree.node_count, -1)
        for i in range(tree.node_count):
            if tree.children_left[i] >= 0:
                parents[tree.children_left[i]] = i
            if tree.children_right[i] >= 0:
                parents[tree.children_right[i]] = i
        n = node_id
        while parents[n] >= 0:
            n = parents[n]
            d += 1
        return d

    init_bounds = {
        "x_min": X[:, 0].min() - 1, "x_max": X[:, 0].max() + 1,
        "y_min": X[:, 1].min() - 5, "y_max": X[:, 1].max() + 5,
        "z_min": X[:, 2].min() - 1, "z_max": X[:, 2].max() + 1,
    }
    _walk(0, init_bounds)
    return splits


def create_tree_splits_figure(max_depth=8):
    """Create interactive 3D scatter with depth-slider split planes.

    Returns a Plotly Figure with animation frames for each depth level.
    """
    X, y = generate_loan_data(200)
    feature_names = ["Income ($K)", "Credit Score", "Loan ($K)"]

    # Fit a tree at maximum depth to get all possible splits
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=3,
                                  random_state=42)
    clf.fit(X, y)
    all_splits = extract_splits_from_sklearn(clf, feature_names, X)

    # Depth color map
    depth_colors = [
        COLORS["secondary"],   # depth 0: blue
        COLORS["purple"],      # depth 1: purple
        COLORS["highlight"],   # depth 2: orange
        COLORS["primary"],     # depth 3: accent blue
        COLORS["gray"],        # depth 4+
        "#a78bfa", "#fbbf24", "#6ee7b7", "#f87171", "#60a5fa",
    ]

    # -- Base scatter traces (always visible) --
    fig = go.Figure()

    # No-default points
    mask_0 = y == 0
    fig.add_trace(go.Scatter3d(
        x=X[mask_0, 0], y=X[mask_0, 1], z=X[mask_0, 2],
        mode="markers",
        marker=dict(size=4, color=COLORS["positive"], opacity=0.8,
                    line=dict(width=0.5, color="white")),
        name="No Default",
        hovertemplate=(
            "Income: $%{x:.0f}K<br>"
            "Credit: %{y:.0f}<br>"
            "Loan: $%{z:.0f}K<br>"
            "<b>No Default</b><extra></extra>"
        ),
    ))

    # Default points
    mask_1 = y == 1
    fig.add_trace(go.Scatter3d(
        x=X[mask_1, 0], y=X[mask_1, 1], z=X[mask_1, 2],
        mode="markers",
        marker=dict(size=4, color=COLORS["negative"], opacity=0.8,
                    line=dict(width=0.5, color="white")),
        name="Default",
        hovertemplate=(
            "Income: $%{x:.0f}K<br>"
            "Credit: %{y:.0f}<br>"
            "Loan: $%{z:.0f}K<br>"
            "<b>Default</b><extra></extra>"
        ),
    ))

    n_scatter = 2  # number of always-visible scatter traces

    # -- Add ALL split planes (initially hidden) --
    for i, sp in enumerate(all_splits):
        color = depth_colors[min(sp["depth"], len(depth_colors) - 1)]
        mesh = make_split_plane_mesh(
            axis=["x", "y", "z"][sp["axis"]],
            value=sp["value"],
            bounds=sp["bounds"],
            color=color,
            opacity=0.15,
        )
        mesh.name = f"D{sp['depth']}: {sp['axis_name']} = {sp['value']:.1f}"
        mesh.visible = False
        mesh.showlegend = False
        fig.add_trace(mesh)

    # -- Slider steps: show planes up to each depth --
    steps = []
    for d in range(max_depth + 1):
        visibility = [True] * n_scatter  # scatter always on
        for sp in all_splits:
            visibility.append(sp["depth"] < d)  # show if depth < slider value

        n_planes = sum(1 for sp in all_splits if sp["depth"] < d)
        n_leaves = n_planes + 1  # rough approximation

        step = dict(
            method="update",
            args=[{"visible": visibility}],
            label=str(d),
        )
        steps.append(step)

    sliders = [dict(
        active=3,
        currentvalue=dict(prefix="Tree Depth: ", font=dict(size=14)),
        pad=dict(t=50),
        steps=steps,
    )]

    # Set initial visibility (depth=3)
    for i, sp in enumerate(all_splits):
        fig.data[n_scatter + i].visible = sp["depth"] < 3

    # -- Layout --
    layout = make_3d_layout(
        title="Decision Tree: Splitting Feature Space",
        x_title="Income ($K)",
        y_title="Credit Score",
        z_title="Loan ($K)",
        width=900,
        height=700,
    )
    layout.sliders = sliders
    fig.update_layout(layout)

    return fig


if __name__ == "__main__":
    fig = create_tree_splits_figure(max_depth=8)
    fig.show()
    fig.write_html("tree_splits_3d.html", include_plotlyjs="cdn")
    print("Saved: tree_splits_3d.html")
