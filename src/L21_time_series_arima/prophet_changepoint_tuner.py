"""
Prophet Changepoint Prior Tuner — Demystifying Changepoint Detection

Slider controls changepoint_prior_scale from 0.001 (very smooth) to 0.5 (very flexible).
Gold dashed lines show true structural breaks; red dashed lines show Prophet's detected
changepoints. Students see how the prior strength controls sensitivity.

Course:  Econ 5200 / 3916
Topic:   Lecture 21 — Time Series II: Prophet Changepoint Detection
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Synthetic data with known structural breaks ──────────────────
T = 120  # 10 years monthly
t = np.arange(T, dtype=float)
dates = pd.date_range("2014-01-01", periods=T, freq="MS")

# Piecewise linear trend with 4 known break points
TRUE_BREAKS = [36, 48, 84, 96]  # month indices
TRUE_BREAK_DATES = [dates[b] for b in TRUE_BREAKS]

slopes = [0.05, -0.15, 0.03, -0.20, 0.08]
breakpoints = [0] + TRUE_BREAKS + [T]

trend = np.zeros(T)
for i in range(len(slopes)):
    seg = slice(breakpoints[i], breakpoints[i + 1])
    seg_t = t[seg] - t[breakpoints[i]]
    if i == 0:
        trend[seg] = slopes[i] * seg_t + 5.0
    else:
        trend[seg] = trend[breakpoints[i] - 1] + slopes[i] * seg_t

seasonal = 1.5 * np.sin(2.0 * np.pi * t / 12.0)
noise = np.random.normal(0, 0.5, T)
y = trend + seasonal + noise

# ── Fit Prophet at multiple prior scales ─────────────────────────
SCALES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

df = pd.DataFrame({"ds": dates, "y": y})

results = {}
for scale in SCALES:
    m = Prophet(
        changepoint_prior_scale=scale,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_range=0.9,
    )
    m.fit(df)
    forecast = m.predict(df)
    trend_fitted = forecast["trend"].values

    # Extract significant changepoints
    deltas = m.params["delta"].flatten()
    n_cp = len(m.changepoints)
    if n_cp > 0:
        delta_vals = deltas[:n_cp]
        threshold = 0.01 * (np.max(y) - np.min(y))
        sig_mask = np.abs(delta_vals) > threshold
        sig_cp_dates = m.changepoints[sig_mask]
    else:
        sig_cp_dates = pd.DatetimeIndex([])

    results[scale] = {
        "trend": trend_fitted,
        "changepoint_dates": sig_cp_dates,
        "n_detected": len(sig_cp_dates),
    }

# ── Build figure ─────────────────────────────────────────────────
fig = go.Figure()

y_min, y_max = y.min() - 1, y.max() + 1

# (Always visible) Original data — gray scatter
fig.add_trace(go.Scatter(
    x=dates,
    y=y,
    mode="markers",
    marker=dict(color=COLORS["gray"], size=4, opacity=0.5),
    name="Observed",
    hovertemplate="%{x|%b %Y}<br>y = %{y:.2f}<extra></extra>",
))

# (Always visible) True break lines — gold dotted
true_x, true_y = [], []
for bd in TRUE_BREAK_DATES:
    true_x += [bd, bd, None]
    true_y += [y_min, y_max, None]

fig.add_trace(go.Scatter(
    x=true_x,
    y=true_y,
    mode="lines",
    line=dict(color=COLORS["gold"], dash="dot", width=2),
    name="True breaks",
    hoverinfo="skip",
))

# Per-scale traces: trend line + detected changepoint lines
TRACES_PER_SCALE = 2
for i, scale in enumerate(SCALES):
    vis = (i == 3)  # default to 0.05 (Prophet's default)
    res = results[scale]

    # Trend line
    fig.add_trace(go.Scatter(
        x=dates,
        y=res["trend"],
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name="Prophet trend",
        visible=vis,
        hovertemplate="%{x|%b %Y}<br>trend = %{y:.2f}<extra></extra>",
    ))

    # Detected changepoints — red dashed
    cp_x, cp_y = [], []
    for cp_date in res["changepoint_dates"]:
        cp_x += [cp_date, cp_date, None]
        cp_y += [y_min, y_max, None]

    fig.add_trace(go.Scatter(
        x=cp_x if len(cp_x) > 0 else [None],
        y=cp_y if len(cp_y) > 0 else [None],
        mode="lines",
        line=dict(color=COLORS["negative"], dash="dash", width=1.5),
        name="Detected changepoints",
        visible=vis,
        hoverinfo="skip",
    ))

# ── Slider ──────────────────────────────────────────────────────
BASE_TRACES = 2  # observed scatter + true breaks
steps = []
for i, scale in enumerate(SCALES):
    res = results[scale]
    visibility = [True, True] + [False] * (len(SCALES) * TRACES_PER_SCALE)
    visibility[BASE_TRACES + i * TRACES_PER_SCALE] = True
    visibility[BASE_TRACES + i * TRACES_PER_SCALE + 1] = True

    steps.append(dict(
        method="update",
        args=[
            {"visible": visibility},
            {"title.text": (
                f"<b>Prophet Changepoint Detection</b><br>"
                f"<span style='font-size:13px; color:{COLORS['gray']}'>"
                f"changepoint_prior_scale = {scale} · "
                f"{res['n_detected']} changepoints detected · "
                f"Gold = true structural breaks</span>"
            )},
        ],
        label=str(scale),
    ))

# ── Layout ──────────────────────────────────────────────────────
default_res = results[SCALES[3]]  # 0.05
fig.update_layout(
    title=dict(text=(
        f"<b>Prophet Changepoint Detection</b><br>"
        f"<span style='font-size:13px; color:{COLORS['gray']}'>"
        f"changepoint_prior_scale = {SCALES[3]} · "
        f"{default_res['n_detected']} changepoints detected · "
        f"Gold = true structural breaks</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    width=950,
    height=550,
    margin=dict(l=60, r=40, t=95, b=80),
    xaxis=dict(title="Date"),
    yaxis=dict(title="Value", range=[y_min, y_max]),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)",
    ),
    sliders=[dict(
        active=3,  # default to 0.05
        currentvalue=dict(prefix="changepoint_prior_scale: "),
        pad=dict(t=40),
        steps=steps,
    )],
)

# ── Export ──────────────────────────────────────────────────────
if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "prophet_changepoint_tuner.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved → {out_path}")
