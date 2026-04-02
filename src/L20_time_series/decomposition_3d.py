"""
Time Series Decomposition 3D Stack
===================================
Visualizes a synthetic time series decomposed into trend, seasonal, and
residual components as stacked 3D ribbons.

Slider: smoothing window (3, 6, 12, 24 months) recomputes the moving-average trend.
Toggle: show/hide each component via legend clicks.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go

np.random.seed(42)

# ------------------------------------------------------------------
# Data generation: 5 years monthly (60 points)
# ------------------------------------------------------------------
T = 60
t = np.arange(1, T + 1, dtype=float)

trend_true = 0.05 * t + 2.0
seasonal_true = 1.5 * np.sin(2.0 * np.pi * t / 12.0)
noise = np.random.normal(0, 0.3, T)
original = trend_true + seasonal_true + noise

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
COMPONENT_COLORS = {
    "Original": "#1a1c1c",
    "Trend": COLORS["secondary"],      # #3b82f6
    "Seasonal": COLORS["highlight"],    # #f97316
    "Residual": COLORS["gray"],         # #6b7280
}

Y_POSITIONS = {"Original": 0, "Trend": 1, "Seasonal": 2, "Residual": 3}


def moving_average(x, w):
    """Centred moving average with edge padding."""
    kernel = np.ones(w) / w
    padded = np.pad(x, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def decompose(window):
    """Return (trend, seasonal, residual) for a given smoothing window."""
    trend = moving_average(original, window)
    detrended = original - trend
    # Average seasonal pattern then tile
    period = 12
    seasonal = np.zeros(T)
    for m in range(period):
        seasonal[m::period] = np.mean(detrended[m::period])
    residual = original - trend - seasonal
    return trend, seasonal, residual


def make_traces(window, visible=True):
    """Build 4 Scatter3d traces for one smoothing-window setting."""
    trend, seasonal, residual = decompose(window)
    data_map = {
        "Original": original,
        "Trend": trend,
        "Seasonal": seasonal,
        "Residual": residual,
    }
    traces = []
    for name in ["Original", "Trend", "Seasonal", "Residual"]:
        y_pos = np.full(T, Y_POSITIONS[name], dtype=float)
        traces.append(go.Scatter3d(
            x=t, y=y_pos, z=data_map[name],
            mode="lines",
            line=dict(color=COMPONENT_COLORS[name], width=4),
            name=name,
            visible=visible,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Month: %{x}<br>"
                "Value: %{z:.2f}<extra></extra>"
            ),
        ))
    return traces


# ------------------------------------------------------------------
# Build figure with slider frames
# ------------------------------------------------------------------
WINDOWS = [3, 6, 12, 24]

fig = go.Figure()

# Add traces for each window (only first window visible)
for i, w in enumerate(WINDOWS):
    for tr in make_traces(w, visible=(i == 0)):
        fig.add_trace(tr)

# Slider steps — each step toggles a group of 4 traces
steps = []
for i, w in enumerate(WINDOWS):
    visibility = [False] * (len(WINDOWS) * 4)
    for j in range(4):
        visibility[i * 4 + j] = True
    steps.append(dict(
        method="update",
        args=[{"visible": visibility}],
        label=f"{w}",
    ))

layout = make_3d_layout(
    title="Time Series Decomposition: Trend + Seasonal + Residual",
    x_title="Month",
    y_title="Component",
    z_title="Value",
    width=900,
    height=700,
)

layout.update(
    title=dict(text=(
        "<b>Time Series Decomposition: Trend + Seasonal + Residual</b><br>"
        "<span style='font-size:13px; color:#6b7280'>Original = Trend + Seasonal + Residual</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=60, b=110),
    annotations=[
        dict(x=0.5, y=-0.15, xref="paper", yref="paper",
             text=(
                 "<span style='color:#1a1c1c'><b>Black</b></span> = original | "
                 "<span style='color:#3b82f6'><b>Blue</b></span> = trend | "
                 "<span style='color:#f97316'><b>Orange</b></span> = seasonal | "
                 "<span style='color:#6b7280'><b>Gray</b></span> = residual"
             ), showarrow=False, font=dict(size=12)),
        dict(x=0.5, y=-0.22, xref="paper", yref="paper",
             text=(
                 "Drag smoothing slider: wider window = smoother trend but may miss short-term changes. "
                 "Large residuals = model is missing something."
             ),
             showarrow=False, font=dict(size=11, color="#6b7280")),
    ],
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix="Smoothing window: ", suffix=" months"),
        pad=dict(t=40),
        steps=steps,
    )],
    scene=dict(
        yaxis=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=["Original", "Trend", "Seasonal", "Residual"],
            title="Component",
            backgroundcolor=COLORS["surface_bg"],
            gridcolor="#e2e2e2",
            showbackground=True,
        ),
        camera=dict(eye=dict(x=1.8, y=-1.4, z=0.9)),
    ),
)

fig.update_layout(layout)

# ------------------------------------------------------------------
if __name__ == "__main__":
    fig.show()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "decomposition_3d.html")
    fig.write_html(out)
    print(f"Saved → {out}")
