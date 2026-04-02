"""
Logistic Regression Decision Boundary
2D scatter of make_moons with contour overlay.
Slider for C values controlling regularization.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# ── Data ────────────────────────────────────────────────────────
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# Meshgrid for contour
margin = 0.5
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                      np.linspace(y_min, y_max, 150))
grid = np.c_[xx.ravel(), yy.ravel()]

# ── C values ────────────────────────────────────────────────────
C_values = [0.01, 0.1, 1, 10, 100]

# ── Build figure ────────────────────────────────────────────────
fig = go.Figure()

for k, C in enumerate(C_values):
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(X, y)
    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    visible = (k == 2)  # default to C=1

    # Contour
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z,
        colorscale=[[0, COLORS["secondary"]], [0.5, "#ffffff"],
                     [1, COLORS["negative"]]],
        showscale=False,
        contours=dict(showlines=True, coloring="heatmap"),
        opacity=0.5,
        visible=visible,
        name=f"C={C}",
        hoverinfo="skip"))

    # Decision boundary line (p=0.5)
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 150),
        y=np.linspace(y_min, y_max, 150),
        z=Z,
        showscale=False,
        contours=dict(
            start=0.5, end=0.5, size=0.1,
            coloring="none",
            showlabels=False),
        line=dict(color=COLORS["gray"], width=2, dash="dash"),
        visible=visible,
        hoverinfo="skip",
        showlegend=False))

# Scatter points (always visible)
for cls, label, color in [(0, "Class 0", COLORS["secondary"]),
                           (1, "Class 1", COLORS["negative"])]:
    mask = y == cls
    fig.add_trace(go.Scatter(
        x=X[mask, 0], y=X[mask, 1],
        mode="markers", name=label,
        marker=dict(color=color, size=7, line=dict(width=1, color="white"))))

# ── Slider ──────────────────────────────────────────────────────
n_per_C = 2  # contour + boundary line per C value
n_scatter = 2

steps = []
for k, C in enumerate(C_values):
    vis = [False] * (len(C_values) * n_per_C) + [True] * n_scatter
    vis[k * n_per_C] = True
    vis[k * n_per_C + 1] = True
    steps.append(dict(
        method="update",
        args=[{"visible": vis},
              {"title": dict(text=(
                  f"<b>Logistic Regression Decision Boundary — C = {C}</b><br>"
                  "<span style='font-size:13px; color:#6b7280'>Decision boundary at P = 0.5</span>"
              ), font=dict(size=16), x=0.5, xanchor="center")}],
        label=str(C)))

fig.update_layout(
    title=dict(text=(
        "<b>Logistic Regression Decision Boundary</b><br>"
        "<span style='font-size:13px; color:#6b7280'>Decision boundary at P = 0.5</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    xaxis=dict(title="Feature 1"),
    yaxis=dict(title="Feature 2", scaleanchor="x"),
    width=800, height=700,
    margin=dict(l=60, r=40, t=80, b=80),
        sliders=[dict(
        active=2,
        currentvalue=dict(prefix="C = "),
        pad=dict(t=50),
        steps=steps)])

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "logistic_boundary.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved to {out}")
    fig.show()
