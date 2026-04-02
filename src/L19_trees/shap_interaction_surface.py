"""
Feature Interactions: SHAP Interaction Surface

3D surface of SHAP interaction values between pairs of features,
revealing whether the model captures multiplicative interactions.
Falls back to a finite-difference approximation if the shap package
is not installed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# ── Generate data with a known multiplicative interaction ───────────────
n = 500
X = np.random.randn(n, 5)
# True DGP: y = 0.5*x1 + 0.3*x2 + 0.8*x1*x2 + 0.1*x3 + noise
y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.8 * X[:, 0] * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(n) * 0.3

feature_names = ["x1", "x2", "x3", "x4", "x5"]

# ── Train model ─────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X, y)
preds = model.predict(X)

# ── Compute SHAP interaction values (with fallback) ─────────────────────
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_interactions = explainer.shap_interaction_values(X)  # (n, 5, 5)
    HAS_SHAP = True
except Exception:
    # Fallback: approximate interaction via finite-difference of predictions
    HAS_SHAP = False
    print("shap not available — using finite-difference approximation")

    def approx_interaction(X_data, model, i, j, delta=0.5):
        """Approximate SHAP interaction as f(+,+) - f(+,-) - f(-,+) + f(-,-)."""
        n = len(X_data)
        interactions = np.zeros(n)
        for k in range(n):
            x = X_data[k].copy()
            x_pp = x.copy(); x_pp[i] += delta; x_pp[j] += delta
            x_pm = x.copy(); x_pm[i] += delta; x_pm[j] -= delta
            x_mp = x.copy(); x_mp[i] -= delta; x_mp[j] += delta
            x_mm = x.copy(); x_mm[i] -= delta; x_mm[j] -= delta
            batch = np.array([x_pp, x_pm, x_mp, x_mm])
            preds = model.predict(batch)
            interactions[k] = (preds[0] - preds[1] - preds[2] + preds[3]) / (4 * delta * delta)
        return interactions

    # Pre-compute for all pairs we'll visualize
    shap_interactions = np.zeros((n, 5, 5))
    pairs_to_compute = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    for pi, pj in pairs_to_compute:
        vals = approx_interaction(X, model, pi, pj)
        shap_interactions[:, pi, pj] = vals
        shap_interactions[:, pj, pi] = vals


def build_interaction_surface(fi, fj, grid_n=30):
    """Build a gridded interaction surface for feature pair (fi, fj)."""
    x_range = np.linspace(X[:, fi].min(), X[:, fi].max(), grid_n)
    y_range = np.linspace(X[:, fj].min(), X[:, fj].max(), grid_n)
    Xg, Yg = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(Xg)

    # Bin observations and average interaction values
    interaction_vals = shap_interactions[:, fi, fj]
    for ix in range(grid_n):
        for iy in range(grid_n):
            dx = 0.5 * (x_range[1] - x_range[0]) if grid_n > 1 else 1.0
            dy = 0.5 * (y_range[1] - y_range[0]) if grid_n > 1 else 1.0
            mask = (
                (np.abs(X[:, fi] - x_range[ix]) < dx * 1.5) &
                (np.abs(X[:, fj] - y_range[iy]) < dy * 1.5)
            )
            if mask.sum() > 0:
                Z[iy, ix] = interaction_vals[mask].mean()
            else:
                # Interpolate from nearest points
                dists = (X[:, fi] - x_range[ix])**2 + (X[:, fj] - y_range[iy])**2
                nearest = np.argsort(dists)[:5]
                weights = 1.0 / (dists[nearest] + 1e-8)
                Z[iy, ix] = np.average(interaction_vals[nearest], weights=weights)

    return x_range, y_range, Z


# ── Feature pairs for dropdown ──────────────────────────────────────────
pairs = [
    (0, 1, "x1 vs x2 (strong interaction)"),
    (0, 2, "x1 vs x3 (weak/no interaction)"),
    (1, 2, "x2 vs x3 (weak/no interaction)"),
    (0, 3, "x1 vs x4 (no interaction)"),
    (1, 3, "x2 vs x4 (no interaction)"),
]

# ── Build figure ────────────────────────────────────────────────────────
fig = go.Figure()

for k, (fi, fj, label) in enumerate(pairs):
    x_r, y_r, Z = build_interaction_surface(fi, fj)
    is_default = (k == 0)

    # Surface
    fig.add_trace(go.Surface(
        x=x_r, y=y_r, z=Z,
        colorscale=[[0, COLORS["secondary"]], [0.5, "#f5f5f5"], [1, COLORS["highlight"]]],
        opacity=0.88,
        showscale=is_default,
        colorbar=dict(title="Interaction", len=0.6) if is_default else None,
        name=label,
        visible=is_default,
        hovertemplate=(
            f"{feature_names[fi]}: " + "%{x:.2f}<br>" +
            f"{feature_names[fj]}: " + "%{y:.2f}<br>" +
            "Interaction: %{z:.3f}<extra></extra>"
        )))

    # Scatter of observations
    interaction_vals = shap_interactions[:, fi, fj]
    fig.add_trace(go.Scatter3d(
        x=X[:, fi], y=X[:, fj], z=interaction_vals,
        mode="markers",
        marker=dict(
            size=2.5, color=preds, colorscale="Viridis",
            opacity=0.5, showscale=False),
        name="Observations",
        showlegend=is_default,
        visible=is_default))

# ── Dropdown menu ───────────────────────────────────────────────────────
n_per_pair = 2  # surface + scatter
buttons = []
for k, (fi, fj, label) in enumerate(pairs):
    vis = [False] * (len(pairs) * n_per_pair)
    vis[k * n_per_pair] = True
    vis[k * n_per_pair + 1] = True
    buttons.append(dict(
        method="update",
        args=[{"visible": vis},
              {"scene.xaxis.title": feature_names[fi],
               "scene.yaxis.title": feature_names[fj]}],
        label=label))

# ── Layout ──────────────────────────────────────────────────────────────
layout = make_3d_layout(
    title="Feature Interactions: SHAP Interaction Surface",
    x_title=feature_names[pairs[0][0]],
    y_title=feature_names[pairs[0][1]],
    z_title="SHAP Interaction Value",
    width=950, height=720)
layout.update(
    title=dict(
        text=(
            "<b>Feature Interactions: SHAP Interaction Surface</b><br>"
            "<span style='font-size:13px; color:#6b7280'>"
            "Flat surface = additive features, Curved = interaction effect</span>"
        ),
        font=dict(size=16), x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=60, b=80),
    updatemenus=[dict(
        type="dropdown",
        buttons=buttons,
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="white",
        font=dict(size=11),
        pad=dict(t=10))],
)

fig.update_layout(layout)

# ── Output ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "shap_interaction_surface.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved: {out_path}")
    fig.show()
