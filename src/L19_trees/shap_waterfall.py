"""
SHAP Waterfall Chart — Feature Attribution for Individual Predictions

Shows how each feature pushes a loan prediction toward or away from default.
Red bars = pushes toward default, Blue bars = pushes away.
Dropdown to select different borrowers.

Course:  Econ 5200 / 3916
Topic:   Lecture 19 — Tree-Based Models: SHAP Interpretability
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
                 "Empl. Years", "Debt Ratio"]

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


def fmt_val(name, val):
    """Format feature value for y-axis label."""
    if name in ("Income", "Loan Amount"):
        return f"${val:,.0f}"
    elif name == "Credit Score":
        return f"{val:.0f}"
    elif name == "Empl. Years":
        return f"{val:.1f} yrs"
    elif name == "Debt Ratio":
        return f"{val:.0%}"
    return f"{val:.1f}"


def build_obs_data(obs_idx):
    """Prepare sorted SHAP data for one borrower."""
    sv = shap_matrix[obs_idx]
    si = np.argsort(np.abs(sv))  # ascending by |SHAP|
    shap_sorted = sv[si]
    raw_sorted = X[obs_idx, si]
    y_labels = [f"{feature_names[int(si[k])]} ({fmt_val(feature_names[int(si[k])], raw_sorted[k])})"
                for k in range(len(si))]
    colors = [COLORS["negative"] if v > 0 else COLORS["secondary"] for v in shap_sorted]
    output = base_value + sv.sum()
    pred = "Default" if clf.predict(X[obs_idx:obs_idx + 1])[0] == 1 else "No Default"
    return y_labels, shap_sorted, colors, output, pred


# ── Build figure ──────────────────────────────────────────────
y0, s0, c0, out0, pred0 = build_obs_data(0)

fig = go.Figure()

fig.add_trace(go.Bar(
    y=y0,
    x=s0.tolist(),
    orientation="h",
    marker=dict(color=c0, line=dict(width=0.5, color="white")),
    hovertemplate="%{y}<br>SHAP = %{x:+.4f}<extra></extra>",
    showlegend=False,
))

fig.add_vline(x=0, line=dict(color=COLORS["gray"], dash="dash", width=1))

# ── Dropdown — right side, not overlapping title ───────────────
buttons = []
for obs in range(10):
    yl, ss, cc, out, pred = build_obs_data(obs)
    bnum = obs + 1  # 1-indexed for students
    buttons.append(dict(
        label=f"Borrower {bnum} ({pred})",
        method="update",
        args=[
            {"y": [yl], "x": [ss.tolist()],
             "marker": [dict(color=cc, line=dict(width=0.5, color="white"))]},
            {"title.text": (
                f"<b>SHAP Feature Attribution</b> — Borrower {bnum}<br>"
                f"<span style='font-size:13px; color:{COLORS['gray']}'>"
                f"Base rate: {base_value:.3f} | Output: {out:.3f} | "
                f"Prediction: <b>{pred}</b></span>"
            )},
        ],
    ))

fig.update_layout(
    title=dict(text=(
        f"<b>SHAP Feature Attribution</b> — Borrower 1<br>"
        f"<span style='font-size:13px; color:{COLORS['gray']}'>"
        f"Base rate: {base_value:.3f} | Output: {out0:.3f} | "
        f"Prediction: <b>{pred0}</b></span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    xaxis=dict(
        title="SHAP Value — Red = pushes toward default, Blue = pushes away",
        zeroline=True, zerolinecolor=COLORS["gray"],
    ),
    yaxis=dict(title="", automargin=True),
    width=900,
    height=500,
    margin=dict(l=200, r=40, t=90, b=60),
    updatemenus=[dict(
        type="dropdown", direction="down",
        x=1.0, xanchor="right",
        y=1.02, yanchor="bottom",
        buttons=buttons,
        bgcolor="white",
        bordercolor=COLORS["gray"],
    )],
)

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "shap_waterfall.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved to {out_path}")
    fig.show()
