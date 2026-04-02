"""
Cost-Sensitive Classification: Finding the Optimal Threshold

3D surface showing how total misclassification cost varies with
decision threshold and class imbalance ratio. A slider for
FP/FN cost ratio reshapes the landscape.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# ── Generate base dataset (imbalanced) ──────────────────────────────────
X_base, y_base = make_classification(
    n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
    weights=[0.5, 0.5], flip_y=0.05, random_state=42,
)

# Train one model on the balanced data — we'll resample test sets
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_base, y_base)
proba = model.predict_proba(X_base)[:, 1]

# ── Grid axes ───────────────────────────────────────────────────────────
thresholds = np.linspace(0.02, 0.98, 50)
imbalance_ratios = np.linspace(0.01, 0.50, 20)  # fraction of positives

# ── Cost ratios to iterate over (FP_cost / FN_cost) ────────────────────
cost_ratios = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def compute_cost_surface(fp_cost, fn_cost):
    """Compute cost surface and find the optimal threshold."""
    Z = np.zeros((len(imbalance_ratios), len(thresholds)))
    rng = np.random.RandomState(42)

    for i, imb in enumerate(imbalance_ratios):
        # Resample to match imbalance ratio
        idx_pos = np.where(y_base == 1)[0]
        idx_neg = np.where(y_base == 0)[0]
        n_total = 500
        n_pos = max(5, int(n_total * imb))
        n_neg = n_total - n_pos
        sampled_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=True)
        sampled_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=True)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        y_sampled = y_base[sampled]
        p_sampled = proba[sampled]

        for j, thresh in enumerate(thresholds):
            y_pred = (p_sampled >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_sampled, y_pred, labels=[0, 1]
            ).ravel()
            Z[i, j] = fp_cost * fp + fn_cost * fn

    # Normalize to per-sample cost
    Z = Z / 500.0
    # Find optimal point
    opt_idx = np.unravel_index(Z.argmin(), Z.shape)
    return Z, opt_idx


# ── Pre-compute all surfaces ────────────────────────────────────────────
all_surfaces = {}
all_optima = {}
for ratio in cost_ratios:
    fp_cost = ratio
    fn_cost = 1.0
    Z, opt_idx = compute_cost_surface(fp_cost, fn_cost)
    all_surfaces[ratio] = Z
    all_optima[ratio] = opt_idx

# ── Build figure ────────────────────────────────────────────────────────
fig = go.Figure()

for k, ratio in enumerate(cost_ratios):
    Z = all_surfaces[ratio]
    opt_i, opt_j = all_optima[ratio]
    is_default = (ratio == 1.0)

    # Surface
    fig.add_trace(go.Surface(
        x=thresholds, y=imbalance_ratios, z=Z,
        colorscale=[[0, COLORS["positive"]], [0.5, COLORS["gold"]],
                     [1, COLORS["negative"]]],
        opacity=0.9,
        showscale=True if is_default else False,
        colorbar=dict(title="Cost", len=0.6) if is_default else None,
        name=f"FP/FN={ratio}",
        visible=is_default,
        hovertemplate=(
            "Threshold: %{x:.2f}<br>"
            "Imbalance: %{y:.2f}<br>"
            "Cost: %{z:.3f}<extra></extra>"
        ),
    ))

    # Optimal point marker
    fig.add_trace(go.Scatter3d(
        x=[thresholds[opt_j]],
        y=[imbalance_ratios[opt_i]],
        z=[Z[opt_i, opt_j]],
        mode="markers+text",
        marker=dict(size=8, color=COLORS["highlight"], symbol="diamond",
                    line=dict(width=2, color="white")),
        text=[f"Optimal: t={thresholds[opt_j]:.2f}"],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["highlight"]),
        name="Optimal threshold",
        showlegend=is_default,
        visible=is_default,
    ))

# ── Slider ──────────────────────────────────────────────────────────────
n_per_ratio = 2  # surface + optimal marker
total = len(cost_ratios) * n_per_ratio

steps = []
for k, ratio in enumerate(cost_ratios):
    vis = [False] * total
    vis[k * n_per_ratio] = True       # surface
    vis[k * n_per_ratio + 1] = True   # optimal marker
    steps.append(dict(
        method="update",
        args=[{"visible": vis}],
        label=str(ratio),
    ))

sliders = [dict(
    active=cost_ratios.index(1.0),
    currentvalue=dict(prefix="FP/FN cost ratio = ", font=dict(size=14)),
    pad=dict(t=40),
    steps=steps,
)]

# ── Layout ──────────────────────────────────────────────────────────────
layout = make_3d_layout(
    title="Cost-Sensitive Classification: Finding the Optimal Threshold",
    x_title="Decision Threshold",
    y_title="Class Imbalance Ratio (% positive)",
    z_title="Avg. Misclassification Cost",
    width=950, height=720,
)
layout.update(
    title=dict(text=(
        "<b>Cost-Sensitive Classification: Optimal Threshold</b><br>"
        "<span style='font-size:13px; color:#6b7280'>Optimal threshold shifts with cost asymmetry</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=60, b=110),
    sliders=sliders,
    annotations=[
        dict(x=0.5, y=-0.15, xref="paper", yref="paper",
             text="Surface height = total misclassification cost | "
                  "<span style='color:#ef4444'>Diamond</span> = optimal threshold",
             showarrow=False, font=dict(size=12)),
        dict(x=0.5, y=-0.22, xref="paper", yref="paper",
             text="Drag cost ratio slider: when false negatives are expensive, optimal threshold moves left (lower, more conservative).",
             showarrow=False, font=dict(size=11, color="#6b7280")),
    ],
)
layout.scene.camera = dict(eye=dict(x=1.8, y=-1.5, z=1.0))

fig.update_layout(layout)

# ── Output ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "cost_loss_landscape.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved: {out_path}")
    fig.show()
