"""
ACF/PACF Pattern Matcher — Identify ARIMA Orders from Diagnostic Plots

Dropdown selects different data-generating processes (AR, MA, ARMA, White Noise).
Three-panel display: simulated series | ACF | PACF.
Significant lags are color-coded; subtitle explains the diagnostic signature.

Course:  Econ 5200 / 3916
Topic:   Lecture 21 — Time Series II: Box-Jenkins Identification
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

np.random.seed(42)

# ── DGP configurations ──────────────────────────────────────────
T = 300
NLAGS = 20
conf_bound = 1.96 / np.sqrt(T)

DGPS = [
    {
        "name": "AR(1), φ = 0.8",
        "ar": [1, -0.8],
        "ma": [1],
        "rule": "ACF decays exponentially → PACF cuts off after lag 1 → AR(1)",
    },
    {
        "name": "AR(2), φ₁ = 0.5, φ₂ = 0.3",
        "ar": [1, -0.5, -0.3],
        "ma": [1],
        "rule": "ACF decays → PACF cuts off after lag 2 → AR(2)",
    },
    {
        "name": "MA(1), θ = 0.7",
        "ar": [1],
        "ma": [1, 0.7],
        "rule": "ACF cuts off after lag 1 → PACF decays → MA(1)",
    },
    {
        "name": "MA(2), θ₁ = 0.5, θ₂ = 0.3",
        "ar": [1],
        "ma": [1, 0.5, 0.3],
        "rule": "ACF cuts off after lag 2 → PACF decays → MA(2)",
    },
    {
        "name": "ARMA(1,1), φ = 0.6, θ = 0.4",
        "ar": [1, -0.6],
        "ma": [1, 0.4],
        "rule": "Both ACF and PACF decay — compare AIC to choose orders → ARMA(1,1)",
    },
    {
        "name": "White Noise",
        "ar": [1],
        "ma": [1],
        "rule": "Both ACF and PACF near zero — no structure → WN (no model needed)",
    },
]

# ── Generate data and compute ACF/PACF for each DGP ─────────────
dgp_data = []
for dgp in DGPS:
    proc = ArmaProcess(dgp["ar"], dgp["ma"])
    series = proc.generate_sample(nsample=T, burnin=200)

    acf_vals = acf(series, nlags=NLAGS, fft=True)
    pacf_vals = pacf(series, nlags=NLAGS, method="ywm")

    # Skip lag 0 for bar charts (it's always 1.0 for ACF)
    lags = np.arange(1, NLAGS + 1)
    acf_bars = acf_vals[1:]
    pacf_bars = pacf_vals[1:]

    dgp_data.append({
        "series": series,
        "lags": lags,
        "acf": acf_bars,
        "pacf": pacf_bars,
    })

# ── Build figure ─────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Simulated Series", "ACF", "PACF"],
    horizontal_spacing=0.08,
)

# Traces per DGP: 1 (series) + 1 (ACF bars) + 2 (ACF bounds) + 1 (PACF bars) + 2 (PACF bounds) = 7
TRACES_PER_DGP = 7

for i, (dgp, data) in enumerate(zip(DGPS, dgp_data)):
    vis = (i == 0)

    # (1) Series line
    fig.add_trace(go.Scatter(
        x=list(range(T)),
        y=data["series"].tolist(),
        mode="lines",
        line=dict(color=COLORS["secondary"], width=1),
        showlegend=False,
        visible=vis,
        hovertemplate="t=%{x}<br>y=%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # (2) ACF bar chart — color by significance
    acf_colors = [COLORS["highlight"] if abs(v) > conf_bound else COLORS["gray"]
                  for v in data["acf"]]
    fig.add_trace(go.Bar(
        x=data["lags"].tolist(),
        y=data["acf"].tolist(),
        marker=dict(color=acf_colors, line=dict(width=0.5, color="white")),
        showlegend=False,
        visible=vis,
        hovertemplate="Lag %{x}<br>ACF=%{y:.3f}<extra></extra>",
    ), row=1, col=2)

    # (3-4) ACF confidence bounds
    for sign in [1, -1]:
        fig.add_trace(go.Scatter(
            x=[0.5, NLAGS + 0.5],
            y=[sign * conf_bound, sign * conf_bound],
            mode="lines",
            line=dict(color=COLORS["negative"], dash="dash", width=1),
            showlegend=False,
            visible=vis,
            hoverinfo="skip",
        ), row=1, col=2)

    # (5) PACF bar chart — color by significance
    pacf_colors = [COLORS["primary"] if abs(v) > conf_bound else COLORS["gray"]
                   for v in data["pacf"]]
    fig.add_trace(go.Bar(
        x=data["lags"].tolist(),
        y=data["pacf"].tolist(),
        marker=dict(color=pacf_colors, line=dict(width=0.5, color="white")),
        showlegend=False,
        visible=vis,
        hovertemplate="Lag %{x}<br>PACF=%{y:.3f}<extra></extra>",
    ), row=1, col=3)

    # (6-7) PACF confidence bounds
    for sign in [1, -1]:
        fig.add_trace(go.Scatter(
            x=[0.5, NLAGS + 0.5],
            y=[sign * conf_bound, sign * conf_bound],
            mode="lines",
            line=dict(color=COLORS["negative"], dash="dash", width=1),
            showlegend=False,
            visible=vis,
            hoverinfo="skip",
        ), row=1, col=3)


# ── Dropdown ────────────────────────────────────────────────────
buttons = []
for i, dgp in enumerate(DGPS):
    visibility = [False] * (len(DGPS) * TRACES_PER_DGP)
    for j in range(TRACES_PER_DGP):
        visibility[i * TRACES_PER_DGP + j] = True

    buttons.append(dict(
        label=dgp["name"],
        method="update",
        args=[
            {"visible": visibility},
            {"title.text": (
                f"<b>ACF / PACF Pattern Matcher — {dgp['name']}</b><br>"
                f"<span style='font-size:13px; color:{COLORS['gray']}'>"
                f"{dgp['rule']}</span>"
            )},
        ],
    ))

# ── Layout ──────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text=(
        f"<b>ACF / PACF Pattern Matcher — {DGPS[0]['name']}</b><br>"
        f"<span style='font-size:13px; color:{COLORS['gray']}'>"
        f"{DGPS[0]['rule']}</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    width=1000,
    height=420,
    margin=dict(l=50, r=30, t=100, b=50),
    updatemenus=[dict(
        type="dropdown", direction="down",
        x=1.0, xanchor="right",
        y=1.04, yanchor="bottom",
        buttons=buttons,
        bgcolor="white",
        bordercolor=COLORS["gray"],
    )],
)

# Axis labels
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_xaxes(title_text="Lag", row=1, col=2)
fig.update_xaxes(title_text="Lag", row=1, col=3)
fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Correlation", row=1, col=2)
fig.update_yaxes(title_text="Partial Correlation", row=1, col=3)

# Set ACF/PACF y-axis range
fig.update_yaxes(range=[-0.5, 1.05], row=1, col=2)
fig.update_yaxes(range=[-0.5, 1.05], row=1, col=3)

# ── Export ──────────────────────────────────────────────────────
if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "acf_pacf_matcher.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved → {out_path}")
