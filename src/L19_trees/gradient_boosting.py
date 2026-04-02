"""
Gradient Boosting Stage-by-Stage
Subplots: top = data + model prediction, bottom = residuals.
Slider for boosting stage, dropdown for learning rate.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# ── Synthetic 1D regression ─────────────────────────────────────
n = 80
X = np.sort(np.random.uniform(0, 10, n))
y_true_fn = lambda x: np.sin(1.5 * x)
y = y_true_fn(X) + np.random.randn(n) * 0.3

X_plot = np.linspace(0, 10, 300)
y_truth = y_true_fn(X_plot)

# ── Pre-compute boosting stages for each learning rate ──────────
learning_rates = [0.05, 0.1, 0.3, 0.5, 1.0]
n_stages = 20

# Store predictions and residuals for each (lr, stage)
all_results = {}
for lr in learning_rates:
    preds = np.full(n, y.mean())
    preds_plot = np.full(300, y.mean())
    stage_data = []

    for s in range(n_stages):
        residuals = y - preds
        tree = DecisionTreeRegressor(max_depth=2, random_state=s)
        tree.fit(X.reshape(-1, 1), residuals)
        update = tree.predict(X.reshape(-1, 1))
        update_plot = tree.predict(X_plot.reshape(-1, 1))
        preds = preds + lr * update
        preds_plot = preds_plot + lr * update_plot
        new_residuals = y - preds
        mse = np.mean(new_residuals ** 2)
        stage_data.append({
            "preds_plot": preds_plot.copy(),
            "residuals": new_residuals.copy(),
            "mse": mse,
        })

    all_results[lr] = stage_data

# ── Build figure ────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=["Data + Model Prediction", "Residuals"],
    vertical_spacing=0.15,
    row_heights=[0.6, 0.4],
)

# Default: lr=0.1, stage=0
default_lr = 0.1
default_stage = 0
d = all_results[default_lr][default_stage]

# Top subplot: scatter + truth + prediction
fig.add_trace(go.Scatter(
    x=X, y=y, mode="markers", name="Training Data",
    marker=dict(color=COLORS["secondary"], size=5, opacity=0.6),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=X_plot, y=y_truth, mode="lines", name="True Function",
    line=dict(color=COLORS["gold"], width=2, dash="dash"),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=X_plot, y=d["preds_plot"], mode="lines", name="Boosted Prediction",
    line=dict(color=COLORS["negative"], width=3),
), row=1, col=1)

# Bottom subplot: residuals
fig.add_trace(go.Scatter(
    x=X, y=d["residuals"], mode="markers", name="Residuals",
    marker=dict(color=COLORS["highlight"], size=5),
    showlegend=True,
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=[0, 10], y=[0, 0], mode="lines",
    line=dict(color=COLORS["gray"], dash="dash", width=1),
    showlegend=False,
), row=2, col=1)

# ── Slider steps (stage) ───────────────────────────────────────
# We'll rebuild traces for the active learning rate via dropdown + slider
# Strategy: use slider for stage, dropdown rebuilds all slider steps

steps = []
for s in range(n_stages):
    d = all_results[default_lr][s]
    steps.append(dict(
        method="update",
        args=[
            {
                "y": [y, y_truth, d["preds_plot"], d["residuals"], [0, 0]],
            },
            {"title": f"Gradient Boosting — Stage {s+1}/{n_stages}, "
                      f"LR={default_lr}, MSE={d['mse']:.4f}"}
        ],
        label=str(s + 1),
    ))

# ── Dropdown for learning rate ──────────────────────────────────
# Each dropdown button resets slider to stage 0 for that LR
buttons = []
for lr in learning_rates:
    d = all_results[lr][0]
    buttons.append(dict(
        label=f"LR = {lr}",
        method="update",
        args=[
            {"y": [y, y_truth, d["preds_plot"], d["residuals"], [0, 0]]},
            {"title": f"Gradient Boosting — Stage 1/{n_stages}, "
                      f"LR={lr}, MSE={d['mse']:.4f}",
             "sliders": [dict(
                 active=0,
                 currentvalue=dict(prefix="Stage: "),
                 pad=dict(t=60),
                 steps=[
                     dict(
                         method="update",
                         args=[
                             {"y": [y, y_truth,
                                    all_results[lr][ss]["preds_plot"],
                                    all_results[lr][ss]["residuals"],
                                    [0, 0]]},
                             {"title": f"Gradient Boosting — Stage {ss+1}/{n_stages}, "
                                       f"LR={lr}, MSE={all_results[lr][ss]['mse']:.4f}"}
                         ],
                         label=str(ss + 1),
                     )
                     for ss in range(n_stages)
                 ],
             )]}
        ],
    ))

fig.update_layout(
    title=f"Gradient Boosting — Stage 1/{n_stages}, LR={default_lr}, "
          f"MSE={all_results[default_lr][0]['mse']:.4f}",
    width=900, height=700,
    xaxis2=dict(title="X"),
    yaxis=dict(title="y"),
    yaxis2=dict(title="Residual"),
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=0.02, xanchor="left",
        y=1.18, yanchor="top",
        buttons=buttons,
    )],
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix="Stage: "),
        pad=dict(t=60),
        steps=steps,
    )],
)

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "gradient_boosting.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved to {out}")
    fig.show()
