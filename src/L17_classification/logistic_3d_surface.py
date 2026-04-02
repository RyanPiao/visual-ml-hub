"""
Logistic Regression: 3D Decision Surface

Interactive 3D visualization of the logistic sigmoid surface over two features.
A slider for regularization C controls surface steepness.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── Generate synthetic 2-feature binary classification data ──────────────
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y=0.1, random_state=42,
)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── Grid for surface ────────────────────────────────────────────────────
grid_n = 60
x1_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, grid_n)
x2_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, grid_n)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

# ── Pre-compute surfaces for each regularization C ──────────────────────
C_values = [0.01, 0.1, 1, 10, 100]
surfaces = {}
for C in C_values:
    model = LogisticRegression(C=C, solver="lbfgs", random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X_grid)[:, 1].reshape(grid_n, grid_n)
    surfaces[C] = probs

# ── Build figure ────────────────────────────────────────────────────────
fig = go.Figure()

# Scatter points for class 0 and class 1 (always visible)
for cls, color, name in [
    (0, COLORS["positive"], "Class 0 (Negative)"),
    (1, COLORS["negative"], "Class 1 (Positive)"),
]:
    mask = y == cls
    # Place points at their predicted probability for the default C=1
    default_model = LogisticRegression(C=1, solver="lbfgs", random_state=42)
    default_model.fit(X, y)
    z_pts = default_model.predict_proba(X[mask])[:, 1]
    fig.add_trace(go.Scatter3d(
        x=X[mask, 0], y=X[mask, 1], z=z_pts,
        mode="markers",
        marker=dict(size=4, color=color, opacity=0.8,
                    line=dict(width=0.5, color="white")),
        name=name,
        showlegend=True,
    ))

# Add a surface for each C value; only C=1 visible by default
for i, C in enumerate(C_values):
    fig.add_trace(go.Surface(
        x=x1_range, y=x2_range, z=surfaces[C],
        colorscale=[[0, COLORS["positive"]], [0.5, "#f5f5f5"], [1, COLORS["negative"]]],
        opacity=0.85,
        showscale=False,
        name=f"C={C}",
        visible=(C == 1),
        hovertemplate="Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<br>P(class=1): %{z:.3f}<extra></extra>",
    ))

# ── Decision boundary plane at P=0.5 (reference) ───────────────────────
fig.add_trace(go.Surface(
    x=x1_range, y=x2_range,
    z=np.full((grid_n, grid_n), 0.5),
    colorscale=[[0, COLORS["gray"]], [1, COLORS["gray"]]],
    opacity=0.15,
    showscale=False,
    name="P=0.5 boundary",
    hoverinfo="skip",
))

# ── Slider for regularization C ────────────────────────────────────────
n_scatter = 2  # two scatter traces (class 0, class 1)
n_surfaces = len(C_values)
n_extra = 1  # the P=0.5 plane
total_traces = n_scatter + n_surfaces + n_extra

steps = []
for i, C in enumerate(C_values):
    visibility = [True] * n_scatter  # scatter always visible
    for j in range(n_surfaces):
        visibility.append(j == i)
    visibility.append(True)  # P=0.5 plane always visible

    label = "Steep = confident" if C >= 10 else ("Flat = uncertain" if C <= 0.1 else f"C={C}")
    steps.append(dict(
        method="update",
        args=[{"visible": visibility}],
        label=str(C),
    ))

sliders = [dict(
    active=2,  # C=1 is index 2
    currentvalue=dict(prefix="Regularization C = ", font=dict(size=14)),
    pad=dict(t=40),
    steps=steps,
)]

# ── Annotations ─────────────────────────────────────────────────────────
layout = make_3d_layout(
    title="Logistic Regression: 3D Decision Surface",
    x_title="Feature 1",
    y_title="Feature 2",
    z_title="P(class = 1)",
    width=900, height=700,
)
layout.update(
    title=dict(text=(
        "<b>Logistic Regression: 3D Decision Surface</b><br>"
        "<span style='font-size:13px; color:#6b7280'>Adjust C slider — steep surface = confident, flat = uncertain</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=60, b=110),
    sliders=sliders,
    annotations=[
        dict(x=0.5, y=-0.15, xref="paper", yref="paper",
             text="<span style='color:#10b981'><b>Green</b></span> = no default | "
                  "<span style='color:#ef4444'><b>Red</b></span> = default | Surface = P(default)",
             showarrow=False, font=dict(size=12)),
        dict(x=0.5, y=-0.22, xref="paper", yref="paper",
             text="Rotate to see the S-shaped sigmoid surface. Higher C = less regularization = steeper, more complex boundary.",
             showarrow=False, font=dict(size=11, color="#6b7280")),
    ],
)
layout.scene.zaxis.update(range=[0, 1])

fig.update_layout(layout)

# ── Output ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "logistic_3d_surface.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved: {out_path}")
    fig.show()
