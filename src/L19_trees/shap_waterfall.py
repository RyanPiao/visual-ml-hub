"""
SHAP Waterfall Chart — Feature Attribution for Individual Predictions

Shows how each feature pushes a prediction toward or away from default.
Red bars = pushes toward default, Blue bars = pushes away.
Dropdown to select different observations.

Course:  Econ 5200 / 3916
Topic:   Lecture 19 — Tree-Based Models: SHAP Interpretability
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# ── Synthetic loan data ─────────────────────────────────────────
n = 300
feature_names = ["Income", "Credit Score", "Loan Amount",
                 "Employment Years", "Debt Ratio"]

rng = np.random.RandomState(42)
income = rng.lognormal(10.5, 0.5, n)
credit = rng.normal(700, 80, n).clip(300, 850)
loan_amount = rng.lognormal(10, 0.8, n)
employment = rng.exponential(5, n).clip(0, 40)
debt_ratio = rng.beta(2, 5, n)

X = np.column_stack([income, credit, loan_amount, employment, debt_ratio])
logit = (-0.00005 * income + 0.008 * (700 - credit) +
         0.00003 * loan_amount - 0.05 * employment + 3 * debt_ratio - 1)
prob = 1 / (1 + np.exp(-logit))
y = (rng.rand(n) < prob).astype(int)

clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X, y)

# ── Compute SHAP values ────────────────────────────────────────
try:
    import shap
    explainer = shap.TreeExplainer(clf)
    shap_values_all = explainer.shap_values(X[:10])
    if isinstance(shap_values_all, list):
        shap_matrix = shap_values_all[1]
    elif shap_values_all.ndim == 3:
        shap_matrix = shap_values_all[:, :, 1]
    else:
        shap_matrix = shap_values_all
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
except ImportError:
    base_value = clf.predict_proba(X)[:, 1].mean()
    shap_matrix = np.zeros((10, 5))
    for i in range(10):
        pred_i = clf.predict_proba(X[i:i + 1])[:, 1][0]
        for j in range(5):
            X_perm = X[i:i + 1].copy()
            X_perm[0, j] = X[:, j].mean()
            pred_perm = clf.predict_proba(X_perm)[:, 1][0]
            shap_matrix[i, j] = pred_i - pred_perm


def build_obs_data(obs_idx):
    """Prepare sorted SHAP data for one observation."""
    sv = shap_matrix[obs_idx]
    si = np.argsort(np.abs(sv))  # ascending by magnitude
    shap_sorted = sv[si]
    raw_sorted = X[obs_idx, si]
    features = [f"{feature_names[int(si[k])]} = {raw_sorted[k]:,.0f}" if raw_sorted[k] > 100
                else f"{feature_names[int(si[k])]} = {raw_sorted[k]:.2f}"
                for k in range(len(si))]
    colors = [COLORS["negative"] if v > 0 else COLORS["secondary"] for v in shap_sorted]
    labels = [f" {v:+.3f} " for v in shap_sorted]
    output = base_value + sv.sum()
    pred = "Default" if clf.predict(X[obs_idx:obs_idx + 1])[0] == 1 else "No Default"
    return features, shap_sorted, colors, labels, output, pred


# ── Initial observation ────────────────────────────────────────
features_0, shap_0, colors_0, labels_0, output_0, pred_0 = build_obs_data(0)

fig = go.Figure()

# Bar trace — SHAP values
fig.add_trace(go.Bar(
    y=features_0,
    x=shap_0.tolist(),
    orientation="h",
    marker=dict(color=colors_0, line=dict(width=0.5, color="white")),
    text=labels_0,
    textposition="outside",
    textfont=dict(size=11),
    hovertemplate="%{y}<br>SHAP = %{x:.4f}<extra></extra>",
    showlegend=False,
    name="SHAP Values",
))

# Zero reference line
fig.add_vline(x=0, line=dict(color=COLORS["gray"], dash="dash", width=1))

# ── Dropdown buttons ────────────────────────────────────────────
buttons = []
for obs in range(10):
    f, s, c, lab, out, pred = build_obs_data(obs)
    buttons.append(dict(
        label=f"Obs {obs} ({pred})",
        method="update",
        args=[
            {"y": [f], "x": [s.tolist()],
             "marker": [dict(color=c, line=dict(width=0.5, color="white"))],
             "text": [lab]},
            {"title.text": (
                f"<b>SHAP Feature Attribution</b> — Observation {obs}<br>"
                f"<span style='font-size:13px; color:{COLORS['gray']}'>"
                f"Base rate: {base_value:.3f}  |  Model output: {out:.3f}  |  "
                f"Prediction: <b>{pred}</b></span>"
            )},
        ],
    ))

# ── Layout ──────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text=(
        f"<b>SHAP Feature Attribution</b> — Observation 0<br>"
        f"<span style='font-size:13px; color:{COLORS['gray']}'>"
        f"Base rate: {base_value:.3f}  |  Model output: {output_0:.3f}  |  "
        f"Prediction: <b>{pred_0}</b></span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    xaxis=dict(
        title="SHAP Value (impact on default probability)",
        zeroline=True,
        zerolinecolor=COLORS["gray"],
    ),
    yaxis=dict(
        title="",
        automargin=True,  # KEY FIX: auto-expand margin to fit labels
    ),
    width=950,
    height=580,
    margin=dict(l=150, r=40, t=100, b=120),  # generous left margin for labels
    updatemenus=[dict(
        type="dropdown", direction="down",
        x=0.0, xanchor="left", y=1.18, yanchor="top",
        buttons=buttons,
        bgcolor="white",
        bordercolor=COLORS["gray"],
    )],
    annotations=[
        # Legend explanation
        dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=(
                "<span style='color:#ef4444'><b>Red</b></span> = pushes toward default  |  "
                "<span style='color:#3b82f6'><b>Blue</b></span> = pushes away from default"
            ),
            showarrow=False, font=dict(size=12),
        ),
        # Context explanation
        dict(
            x=0.5, y=-0.25, xref="paper", yref="paper",
            text=(
                "Each bar shows how much one feature shifts the prediction from the base rate. "
                "Longer bars = stronger influence."
            ),
            showarrow=False, font=dict(size=11, color=COLORS["gray"]),
        ),
    ],
)

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "shap_waterfall.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved to {out_path}")
    fig.show()
