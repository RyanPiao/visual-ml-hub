"""
Forecast Comparison Dashboard — SARIMA vs Prophet vs Seasonal Naive

Slider steps through 5 expanding-window backtest folds + a summary view.
Each fold shows shaded training region, actual test values, and three
forecast lines with MASE scores in the subtitle.

Course:  Econ 5200 / 3916
Topic:   Lecture 21 — Time Series II: Forecast Evaluation & Backtesting
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Synthetic monthly retail sales ───────────────────────────────
T = 84  # 7 years
t = np.arange(T, dtype=float)
dates = pd.date_range("2017-01-01", periods=T, freq="MS")

trend = 100 + 0.3 * t
seasonal = 15 * np.sin(2.0 * np.pi * t / 12.0)
noise = np.random.normal(0, 3, T)
y = trend + seasonal + noise
series = pd.Series(y, index=dates)

# ── Expanding-window backtesting (5 folds) ───────────────────────
MIN_TRAIN = 36   # first 3 years minimum
HORIZON = 6      # 6-month forecast horizon
STEP = 6         # slide by 6 months

folds = []
for fold_i in range(5):
    train_end = MIN_TRAIN + fold_i * STEP
    test_end = min(train_end + HORIZON, T)
    if test_end > T:
        break

    train = series.iloc[:train_end]
    test = series.iloc[train_end:test_end]
    h = len(test)

    # --- SARIMA ---
    try:
        sarima_model = SARIMAX(
            train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        sarima_fit = sarima_model.fit(disp=False, maxiter=200)
        sarima_pred = sarima_fit.forecast(steps=h)
    except Exception:
        sarima_pred = pd.Series(train.values[-12:h] if h <= 12 else
                                np.tile(train.values[-12:], 2)[:h],
                                index=test.index)

    # --- Prophet ---
    try:
        pdf = pd.DataFrame({"ds": train.index, "y": train.values})
        m = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m.fit(pdf)
        future = m.make_future_dataframe(periods=h, freq="MS")
        prophet_fc = m.predict(future)
        prophet_pred = prophet_fc["yhat"].iloc[-h:].values
    except Exception:
        prophet_pred = train.values[-12:][:h] if h <= 12 else \
            np.tile(train.values[-12:], 2)[:h]

    # --- Seasonal Naive ---
    naive_pred = train.values[-12:][:h] if h <= 12 else \
        np.tile(train.values[-12:], 2)[:h]

    # --- MASE ---
    scale = np.mean(np.abs(train.values[12:] - train.values[:-12]))
    if scale < 1e-8:
        scale = 1.0

    mase_sarima = np.mean(np.abs(test.values - sarima_pred.values)) / scale
    mase_prophet = np.mean(np.abs(test.values - prophet_pred)) / scale
    mase_naive = np.mean(np.abs(test.values - naive_pred)) / scale

    folds.append({
        "train_dates": train.index,
        "test_dates": test.index,
        "train_vals": train.values,
        "test_vals": test.values,
        "sarima": sarima_pred.values if hasattr(sarima_pred, 'values') else sarima_pred,
        "prophet": prophet_pred,
        "naive": naive_pred,
        "mase_sarima": mase_sarima,
        "mase_prophet": mase_prophet,
        "mase_naive": mase_naive,
    })

N_FOLDS = len(folds)

# ── Build figure ─────────────────────────────────────────────────
fig = go.Figure()

# Traces per fold: training shade, actual test, SARIMA, Prophet, Naive = 5
TRACES_PER_FOLD = 5

for fi, fold in enumerate(folds):
    vis = (fi == 0)

    # (1) Training region — light shaded area
    fig.add_trace(go.Scatter(
        x=list(fold["train_dates"]) + list(fold["train_dates"][::-1]),
        y=list(fold["train_vals"]) + [0] * len(fold["train_vals"]),
        fill="toself",
        fillcolor="rgba(147, 197, 253, 0.15)",
        line=dict(width=0),
        mode="lines",
        name="Training data",
        visible=vis,
        showlegend=True,
        hoverinfo="skip",
    ))

    # (2) Actual test values
    fig.add_trace(go.Scatter(
        x=list(fold["test_dates"]),
        y=fold["test_vals"].tolist(),
        mode="lines+markers",
        line=dict(color=COLORS["text"], width=2),
        marker=dict(size=6),
        name="Actual",
        visible=vis,
        hovertemplate="%{x|%b %Y}<br>Actual = %{y:.1f}<extra></extra>",
    ))

    # (3) SARIMA forecast
    fig.add_trace(go.Scatter(
        x=list(fold["test_dates"]),
        y=fold["sarima"].tolist(),
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=5, symbol="diamond"),
        name="SARIMA",
        visible=vis,
        hovertemplate="%{x|%b %Y}<br>SARIMA = %{y:.1f}<extra></extra>",
    ))

    # (4) Prophet forecast
    fig.add_trace(go.Scatter(
        x=list(fold["test_dates"]),
        y=fold["prophet"].tolist(),
        mode="lines+markers",
        line=dict(color=COLORS["highlight"], width=2),
        marker=dict(size=5, symbol="square"),
        name="Prophet",
        visible=vis,
        hovertemplate="%{x|%b %Y}<br>Prophet = %{y:.1f}<extra></extra>",
    ))

    # (5) Seasonal Naive
    fig.add_trace(go.Scatter(
        x=list(fold["test_dates"]),
        y=fold["naive"].tolist(),
        mode="lines+markers",
        line=dict(color=COLORS["gray"], width=2, dash="dash"),
        marker=dict(size=5, symbol="cross"),
        name="Naive",
        visible=vis,
        hovertemplate="%{x|%b %Y}<br>Naive = %{y:.1f}<extra></extra>",
    ))

# Summary traces (fold index = N_FOLDS): just the full series
fig.add_trace(go.Scatter(
    x=list(dates),
    y=y.tolist(),
    mode="lines",
    line=dict(color=COLORS["secondary"], width=2),
    name="Full series",
    visible=False,
    hovertemplate="%{x|%b %Y}<br>y = %{y:.1f}<extra></extra>",
))

TOTAL_FOLD_TRACES = N_FOLDS * TRACES_PER_FOLD
SUMMARY_TRACES = 1

# ── Slider ──────────────────────────────────────────────────────
avg_sarima = np.mean([f["mase_sarima"] for f in folds])
avg_prophet = np.mean([f["mase_prophet"] for f in folds])
avg_naive = np.mean([f["mase_naive"] for f in folds])

# Determine overall winner
scores = {"SARIMA": avg_sarima, "Prophet": avg_prophet, "Naive": avg_naive}
winner = min(scores, key=scores.get)

steps = []
for fi, fold in enumerate(folds):
    visibility = [False] * (TOTAL_FOLD_TRACES + SUMMARY_TRACES)
    for j in range(TRACES_PER_FOLD):
        visibility[fi * TRACES_PER_FOLD + j] = True

    steps.append(dict(
        method="update",
        args=[
            {"visible": visibility},
            {"title.text": (
                f"<b>Forecast Comparison — Fold {fi + 1}/{N_FOLDS}</b><br>"
                f"<span style='font-size:13px; color:{COLORS['gray']}'>"
                f"MASE — SARIMA: {fold['mase_sarima']:.3f} · "
                f"Prophet: {fold['mase_prophet']:.3f} · "
                f"Naive: {fold['mase_naive']:.3f}</span>"
            )},
        ],
        label=f"Fold {fi + 1}",
    ))

# Summary step
summary_vis = [False] * (TOTAL_FOLD_TRACES + SUMMARY_TRACES)
summary_vis[-1] = True

steps.append(dict(
    method="update",
    args=[
        {"visible": summary_vis},
        {"title.text": (
            f"<b>Forecast Comparison — Summary</b><br>"
            f"<span style='font-size:13px; color:{COLORS['gray']}'>"
            f"Avg MASE — SARIMA: {avg_sarima:.3f} · "
            f"Prophet: {avg_prophet:.3f} · "
            f"Naive: {avg_naive:.3f} · "
            f"Winner: <b>{winner}</b></span>"
        )},
    ],
    label="Summary",
))

# ── Layout ──────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text=(
        f"<b>Forecast Comparison — Fold 1/{N_FOLDS}</b><br>"
        f"<span style='font-size:13px; color:{COLORS['gray']}'>"
        f"MASE — SARIMA: {folds[0]['mase_sarima']:.3f} · "
        f"Prophet: {folds[0]['mase_prophet']:.3f} · "
        f"Naive: {folds[0]['mase_naive']:.3f}</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    width=950,
    height=550,
    margin=dict(l=60, r=40, t=95, b=80),
    xaxis=dict(title="Date"),
    yaxis=dict(title="Retail Sales"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)",
    ),
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix="Backtest: "),
        pad=dict(t=40),
        steps=steps,
    )],
)

# ── Export ──────────────────────────────────────────────────────
if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "forecast_comparison.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved → {out_path}")
    print(f"\nAvg MASE — SARIMA: {avg_sarima:.3f}, Prophet: {avg_prophet:.3f}, "
          f"Naive: {avg_naive:.3f}")
