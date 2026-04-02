"""
SHAP Waterfall Chart
Train RandomForestClassifier on synthetic loan data.
Horizontal bar chart of SHAP values for individual observations.
Dropdown to select observation index (0-9).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# ── Synthetic loan data ─────────────────────────────────────────
n = 300
feature_names = ["Income", "Credit Score", "Loan Amount",
                 "Employment Years", "Debt Ratio"]

income = np.random.lognormal(10.5, 0.5, n)
credit = np.random.normal(700, 80, n).clip(300, 850)
loan_amount = np.random.lognormal(10, 0.8, n)
employment = np.random.exponential(5, n).clip(0, 40)
debt_ratio = np.random.beta(2, 5, n)

X = np.column_stack([income, credit, loan_amount, employment, debt_ratio])

# Generate labels: higher default prob with low credit, high debt, high loan
logit = (-0.00005 * income + 0.008 * (700 - credit) +
         0.00003 * loan_amount - 0.05 * employment + 3 * debt_ratio - 1)
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)

# ── Train model ─────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X, y)

# ── Compute SHAP values using TreeExplainer ─────────────────────
try:
    import shap
    explainer = shap.TreeExplainer(clf)
    shap_values_all = explainer.shap_values(X[:10])
    # Handle different SHAP output formats:
    # Old shap: list of 2 arrays, each (n, features)
    # New shap: single array (n, features, 2)
    if isinstance(shap_values_all, list):
        shap_matrix = shap_values_all[1]  # (10, 5)
    elif shap_values_all.ndim == 3:
        shap_matrix = shap_values_all[:, :, 1]  # (10, 5) — class 1
    else:
        shap_matrix = shap_values_all
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
except ImportError:
    # Fallback: approximate SHAP with permutation-based approach
    print("shap not installed; using permutation-importance approximation.")
    from sklearn.inspection import permutation_importance
    base_value = clf.predict_proba(X)[:, 1].mean()
    shap_matrix = np.zeros((10, 5))
    for i in range(10):
        pred_i = clf.predict_proba(X[i:i+1])[:, 1][0]
        for j in range(5):
            X_perm = X[i:i+1].copy()
            orig_val = X_perm[0, j]
            # Replace with column mean and measure change
            X_perm[0, j] = X[:, j].mean()
            pred_perm = clf.predict_proba(X_perm)[:, 1][0]
            shap_matrix[i, j] = pred_i - pred_perm

# ── Build figure with dropdown for observations 0-9 ─────────────
fig = go.Figure()

# Add bars for observation 0 initially
obs_idx = 0
shap_vals = shap_matrix[obs_idx]
sorted_idx = np.argsort(np.abs(shap_vals))  # ascending abs value (numpy array)
sorted_features = [feature_names[int(i)] for i in sorted_idx]
sorted_shap = shap_vals[sorted_idx]
sorted_raw = X[obs_idx, sorted_idx]

bar_colors = [COLORS["negative"] if v > 0 else COLORS["secondary"]
              for v in sorted_shap]

fig.add_trace(go.Bar(
    y=sorted_features,
    x=sorted_shap,
    orientation="h",
    marker=dict(color=bar_colors, line=dict(width=1, color="white")),
    text=[f"{v:+.3f} ({feature_names[sorted_idx[k]]}={sorted_raw[k]:.1f})"
          for k, v in enumerate(sorted_shap)],
    textposition="auto",
    hovertemplate="%{y}: SHAP=%{x:.3f}<extra></extra>",
))

# Base value line
model_output = base_value + shap_vals.sum()
fig.add_trace(go.Scatter(
    x=[0, 0], y=[sorted_features[0], sorted_features[-1]],
    mode="lines",
    line=dict(color=COLORS["gray"], dash="dash", width=1),
    showlegend=False,
))

# ── Dropdown buttons ────────────────────────────────────────────
buttons = []
for obs in range(10):
    sv = shap_matrix[obs]
    si = np.argsort(np.abs(sv))
    sf = [feature_names[int(i)] for i in si]
    ss = sv[si]
    sr = X[obs, si]
    bc = [COLORS["negative"] if v > 0 else COLORS["secondary"] for v in ss]
    mo = base_value + sv.sum()
    pred_label = "Default" if clf.predict(X[obs:obs+1])[0] == 1 else "No Default"

    buttons.append(dict(
        label=f"Obs {obs} ({pred_label})",
        method="update",
        args=[
            {
                "y": [sf, [sf[0], sf[-1]]],
                "x": [ss.tolist(), [0, 0]],
                "marker": [dict(color=bc, line=dict(width=1, color="white")), None],
                "text": [[f"{v:+.3f} ({feature_names[si[k]]}={sr[k]:.1f})"
                          for k, v in enumerate(ss)], None],
            },
            {
                "title": (f"SHAP Waterfall — Obs {obs} | "
                          f"Base: {base_value:.3f} | "
                          f"Output: {mo:.3f} | "
                          f"Pred: {pred_label}"),
                "annotations": [dict(
                    x=0, y=-0.12, xref="paper", yref="paper",
                    text=(f"<b>Red</b> = pushes toward default | "
                          f"<b>Blue</b> = pushes away from default | "
                          f"Base value: {base_value:.3f}"),
                    showarrow=False, font=dict(size=11),
                )],
            }
        ],
    ))

pred0 = "Default" if clf.predict(X[0:1])[0] == 1 else "No Default"
mo0 = base_value + shap_matrix[0].sum()

fig.update_layout(
    title=f"SHAP Waterfall — Obs 0 | Base: {base_value:.3f} | "
          f"Output: {mo0:.3f} | Pred: {pred0}",
    xaxis=dict(title="SHAP Value (impact on default probability)"),
    yaxis=dict(title=""),
    width=900, height=550,
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=0.02, xanchor="left",
        y=1.15, yanchor="top",
        buttons=buttons,
    )],
    annotations=[dict(
        x=0, y=-0.12, xref="paper", yref="paper",
        text=(f"<b>Red</b> = pushes toward default | "
              f"<b>Blue</b> = pushes away from default | "
              f"Base value: {base_value:.3f}"),
        showarrow=False, font=dict(size=11),
    )],
)

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "shap_waterfall.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved to {out}")
    fig.show()
