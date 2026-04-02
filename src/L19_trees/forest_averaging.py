"""
Interactive Random Forest Averaging — Plotly 3D Surface

Shows how averaging many decision trees smooths the prediction surface.
- Gold surface = true function (ground truth)
- Blue surface = ensemble average (changes with slider)
- Slider controls B (number of trees): 1 → 100

Students can: rotate both surfaces, drag slider to see smoothing effect.

Usage:
    python forest_averaging.py          # opens in browser
    # or import:
    from forest_averaging import create_forest_figure
    fig = create_forest_figure()
    fig.show()

    # Export for slides
    fig.write_html("forest_averaging.html", include_plotlyjs="cdn")

Course:  Econ 5200 / 3916
Topic:   Lecture 19 — Random Forests: Variance Reduction via Ensemble
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

np.random.seed(42)

# Grid for surface
GRID_N = 40
X1_RANGE = (0, 10)
X2_RANGE = (0, 10)


def true_function(x1, x2):
    """Smooth ground truth: f(income, credit_score)."""
    return (
        3.0
        + 1.5 * np.sin(0.7 * x1)
        + 1.0 * np.cos(0.5 * x2)
        - 0.2 * x1 * x2 / 10
        + 0.8 * np.exp(-((x1 - 5) ** 2 + (x2 - 5) ** 2) / 8)
    )


def build_bootstrap_trees(n_trees=100, n_train=150, max_depth=6, noise=0.8):
    """Train n_trees on bootstrap samples of noisy data."""
    rng = np.random.RandomState(42)

    # Full training set
    x1_train = rng.uniform(*X1_RANGE, n_train)
    x2_train = rng.uniform(*X2_RANGE, n_train)
    y_train = np.array([true_function(a, b) for a, b in zip(x1_train, x2_train)])
    y_train += rng.normal(0, noise, n_train)
    X_train = np.column_stack([x1_train, x2_train])

    # Grid for predictions
    g1 = np.linspace(*X1_RANGE, GRID_N)
    g2 = np.linspace(*X2_RANGE, GRID_N)
    G1, G2 = np.meshgrid(g1, g2)
    X_grid = np.column_stack([G1.ravel(), G2.ravel()])

    # Train each tree on a bootstrap sample
    predictions = []
    for i in range(n_trees):
        idx = rng.choice(n_train, size=n_train, replace=True)
        dt = DecisionTreeRegressor(max_depth=max_depth, random_state=i)
        dt.fit(X_train[idx], y_train[idx])
        pred = dt.predict(X_grid).reshape(GRID_N, GRID_N)
        predictions.append(pred)

    return G1, G2, predictions


def create_forest_figure(n_trees=100):
    """Create figure with slider for number of trees averaged."""
    G1, G2, predictions = build_bootstrap_trees(n_trees=n_trees)

    # True surface
    Z_true = np.array([[true_function(a, b) for a, b in zip(row1, row2)]
                        for row1, row2 in zip(G1, G2)])

    # Pre-compute ensemble averages for each B
    B_values = [1, 2, 3, 5, 10, 20, 50, 100]
    B_values = [b for b in B_values if b <= n_trees]

    ensemble_surfaces = {}
    for b in B_values:
        avg = np.mean(predictions[:b], axis=0)
        ensemble_surfaces[b] = avg

    fig = go.Figure()

    # -- True surface (always visible) --
    fig.add_trace(go.Surface(
        x=G1, y=G2, z=Z_true,
        colorscale=[[0, COLORS["gold"]], [1, COLORS["gold"]]],
        opacity=0.35,
        showscale=False,
        name="True Function",
        hovertemplate="x1: %{x:.1f}<br>x2: %{y:.1f}<br>f(x): %{z:.2f}<extra>Truth</extra>",
    ))

    # -- Ensemble surfaces (one per B, toggle visibility) --
    for i, b in enumerate(B_values):
        Z_ens = ensemble_surfaces[b]
        fig.add_trace(go.Surface(
            x=G1, y=G2, z=Z_ens,
            colorscale=[[0, "#dbeafe"], [1, COLORS["primary"]]],
            opacity=0.7,
            showscale=False,
            visible=(i == 0),  # only B=1 visible initially
            name=f"Ensemble (B={b})",
            hovertemplate="x1: %{x:.1f}<br>x2: %{y:.1f}<br>pred: %{z:.2f}<extra>B=%{text}</extra>",
            text=np.full_like(Z_ens, str(b)),
        ))

    # -- Slider --
    steps = []
    for i, b in enumerate(B_values):
        # True surface always visible, plus one ensemble surface
        visibility = [True] + [j == i for j in range(len(B_values))]

        # Compute RMSE for annotation
        rmse = np.sqrt(np.mean((ensemble_surfaces[b] - Z_true) ** 2))

        step = dict(
            method="update",
            args=[
                {"visible": visibility},
                {"title": dict(
                    text=(
                        f"<b>Random Forest: B = {b} tree{'s' if b > 1 else ''} "
                        f"(RMSE = {rmse:.3f})</b><br>"
                        "<span style='font-size:13px; color:#6b7280'>"
                        "Drag B slider to see variance reduction via averaging</span>"
                    ),
                    font=dict(size=16), x=0.5, xanchor="center",
                )},
            ],
            label=str(b),
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Number of Trees (B): ", font=dict(size=14)),
        pad=dict(t=60),
        steps=steps,
    )]

    # Initial RMSE
    rmse_1 = np.sqrt(np.mean((ensemble_surfaces[B_values[0]] - Z_true) ** 2))

    layout = make_3d_layout(
        title=f"Random Forest: B = 1 tree (RMSE = {rmse_1:.3f})",
        x_title="Feature 1 (Income)",
        y_title="Feature 2 (Credit Score)",
        z_title="Prediction",
        width=900,
        height=700,
    )
    layout.sliders = sliders
    layout.title = dict(
        text=(
            f"<b>Random Forest: B = 1 tree (RMSE = {rmse_1:.3f})</b><br>"
            "<span style='font-size:13px; color:#6b7280'>"
            "Drag B slider to see variance reduction via averaging</span>"
        ),
        font=dict(size=16), x=0.5, xanchor="center",
    )
    layout.margin = dict(l=20, r=20, t=60, b=110)
    layout.annotations = [
        dict(x=0.5, y=-0.15, xref="paper", yref="paper",
             text="<span style='color:#eab308'><b>Gold</b></span> = true function | "
                  "<span style='color:#1357c9'><b>Blue</b></span> = forest prediction",
             showarrow=False, font=dict(size=12)),
        dict(x=0.5, y=-0.22, xref="paper", yref="paper",
             text="Drag B slider: 1 tree = blocky staircase, 100 trees = smooth surface. "
                  "Averaging reduces variance.",
             showarrow=False, font=dict(size=11, color="#6b7280")),
    ]

    fig.update_layout(layout)
    return fig


if __name__ == "__main__":
    fig = create_forest_figure(n_trees=100)
    fig.show()
    fig.write_html("forest_averaging.html", include_plotlyjs="cdn")
    print("Saved: forest_averaging.html")
