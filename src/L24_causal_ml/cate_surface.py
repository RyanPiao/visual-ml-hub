"""
Heterogeneous Treatment Effect (CATE) Surface
===============================================
Synthetic observational data (n=500).  T-learner estimates CATE as a
function of income and age.  3D surface shows estimated CATE, a semi-
transparent gold surface for true CATE, and individual scatter points.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# ------------------------------------------------------------------
# Data generating process
# ------------------------------------------------------------------
N = 500
income = np.random.uniform(20, 100, N)  # thousands
age = np.random.uniform(25, 65, N)

# Propensity: treatment more likely for higher income, younger
propensity = 1.0 / (1.0 + np.exp(-(0.02 * (income - 50) - 0.01 * (age - 45))))
W = np.random.binomial(1, propensity, N)

# True CATE
def true_cate(inc, a):
    return (2.0
            + 0.03 * (inc - 50)
            - 0.05 * (a - 40)
            + 0.001 * (inc - 50) * (a - 40))

tau = true_cate(income, age)

# Outcome
base = 10 + 0.1 * income - 0.05 * age
noise = np.random.normal(0, 1.5, N)
Y = base + W * tau + noise

# ------------------------------------------------------------------
# T-learner: separate models for treated / control
# ------------------------------------------------------------------
X = np.column_stack([income, age])

treated_mask = W == 1
control_mask = W == 0

rf_t = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_c = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

rf_t.fit(X[treated_mask], Y[treated_mask])
rf_c.fit(X[control_mask], Y[control_mask])

# Individual CATE estimates
cate_hat = rf_t.predict(X) - rf_c.predict(X)

# ------------------------------------------------------------------
# Grid for surface
# ------------------------------------------------------------------
GRID = 40
inc_grid = np.linspace(20, 100, GRID)
age_grid = np.linspace(25, 65, GRID)
INC, AGE = np.meshgrid(inc_grid, age_grid)
X_grid = np.column_stack([INC.ravel(), AGE.ravel()])

cate_surface = (rf_t.predict(X_grid) - rf_c.predict(X_grid)).reshape(GRID, GRID)
true_surface = true_cate(INC, AGE)

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig = go.Figure()

# 1. Estimated CATE surface
fig.add_trace(go.Surface(
    x=inc_grid, y=age_grid, z=cate_surface,
    colorscale=[
        [0.0, COLORS["negative"]],
        [0.5, "#fafafa"],
        [1.0, COLORS["positive"]],
    ],
    opacity=0.85,
    name="Estimated CATE",
    showscale=True,
    colorbar=dict(title="CATE", len=0.55, x=1.02),
    hovertemplate=(
        "Income: %{x:.0f}K<br>"
        "Age: %{y:.0f}<br>"
        "Est. CATE: %{z:.2f}<extra>Estimated</extra>"
    )))

# 2. True CATE surface (gold, semi-transparent)
fig.add_trace(go.Surface(
    x=inc_grid, y=age_grid, z=true_surface,
    colorscale=[[0, COLORS["gold"]], [1, COLORS["gold"]]],
    opacity=0.35,
    name="True CATE",
    showscale=False,
    hovertemplate=(
        "Income: %{x:.0f}K<br>"
        "Age: %{y:.0f}<br>"
        "True CATE: %{z:.2f}<extra>True</extra>"
    )))

# 3. Contour at CATE = 0 — simple numpy approach (no skimage needed)
# Walk each row of the grid to find sign changes
for i in range(GRID):
    for j in range(GRID - 1):
        if cate_surface[i, j] * cate_surface[i, j + 1] < 0:  # sign change
            # Linear interpolation for zero crossing
            frac = -cate_surface[i, j] / (cate_surface[i, j + 1] - cate_surface[i, j])
            x0 = inc_grid[j] + frac * (inc_grid[j + 1] - inc_grid[j])
            fig.add_trace(go.Scatter3d(
                x=[x0], y=[age_grid[i]], z=[0],
                mode="markers",
                marker=dict(color="black", size=2),
                showlegend=False, hoverinfo="skip"))
# Add a single legend entry for the contour dots
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None], mode="markers",
    marker=dict(color="black", size=4), name="CATE = 0 boundary"))

# 4. Individual scatter points
marker_colors = np.where(cate_hat >= 0, COLORS["positive"], COLORS["negative"])
fig.add_trace(go.Scatter3d(
    x=income, y=age, z=cate_hat,
    mode="markers",
    marker=dict(
        size=3,
        color=marker_colors,
        opacity=0.6,
        line=dict(width=0.3, color="white")),
    name="Individual CATE est.",
    customdata=np.column_stack([
        np.round(cate_hat, 2).astype(str),
        np.round(tau, 2).astype(str),
        W.astype(str),
    ]),
    hovertemplate=(
        "Income: %{x:.0f}K | Age: %{y:.0f}<br>"
        "Est. CATE: %{customdata[0]}<br>"
        "True CATE: %{customdata[1]}<br>"
        "Treated: %{customdata[2]}<extra></extra>"
    )))

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
layout = make_3d_layout(
    title="Heterogeneous Treatment Effects: Who Benefits Most?",
    x_title="Income ($K)",
    y_title="Age",
    z_title="CATE (treatment effect)",
    width=950,
    height=720)
layout.update(
    title=dict(text=(
        "<b>Heterogeneous Treatment Effects: Who Benefits Most?</b><br>"
        "<span style='font-size:13px; color:#6b7280'>Who benefits most from treatment?</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=60, b=80),
        scene=dict(camera=dict(eye=dict(x=1.6, y=-1.5, z=1.0))),
    legend=dict(orientation="v", x=1.08, y=0.95))
fig.update_layout(layout)

# ------------------------------------------------------------------
if __name__ == "__main__":
    fig.show()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cate_surface.html")
    fig.write_html(out)
    print(f"Saved → {out}")
