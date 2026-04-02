"""
Bootstrap Sampling Visualization
Animated bar chart showing bootstrap samples, OOB observations,
and a summary of OOB frequency.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go

np.random.seed(42)

# ── Data: 20 points ─────────────────────────────────────────────
n = 20
indices = np.arange(n)
values = indices + 1 + np.random.randn(n) * 1.5  # ~1-20 with noise
labels = [f"Obs {i+1}" for i in range(n)]

# ── Generate 5 bootstrap samples ────────────────────────────────
n_bootstrap = 5
boot_samples = []
for b in range(n_bootstrap):
    sample = np.random.choice(n, size=n, replace=True)
    boot_samples.append(sample)

# Track OOB frequency across all 5 samples
oob_count = np.zeros(n, dtype=int)
for sample in boot_samples:
    in_sample = set(sample)
    for i in range(n):
        if i not in in_sample:
            oob_count[i] += 1

# ── Build frames ────────────────────────────────────────────────
frames = []

# Frame 0: original data — all blue
frames.append(go.Frame(
    data=[go.Bar(
        x=labels, y=values,
        marker=dict(color=[COLORS["secondary"]] * n,
                    line=dict(width=1, color="white")),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        opacity=1.0,
    )],
    name="Original",
    layout=go.Layout(title="Bootstrap Sampling — Original Data"),
))

# Frames 1-5: bootstrap samples
for b in range(n_bootstrap):
    sample = boot_samples[b]
    counts = np.bincount(sample, minlength=n)
    colors = []
    texts = []
    opacities = []
    for i in range(n):
        if counts[i] > 0:
            colors.append(COLORS["positive"])
            texts.append(f"x{counts[i]}")
        else:
            colors.append(COLORS["gray"])
            texts.append("OOB")

    bar_opacity = [1.0 if counts[i] > 0 else 0.35 for i in range(n)]

    frames.append(go.Frame(
        data=[go.Bar(
            x=labels, y=values,
            marker=dict(
                color=colors,
                line=dict(width=1, color="white"),
                opacity=bar_opacity,
            ),
            text=texts,
            textposition="outside",
        )],
        name=f"Sample {b+1}",
        layout=go.Layout(
            title=f"Bootstrap Sample {b+1} — "
                  f"{len(set(sample))}/{n} unique observations drawn"
        ),
    ))

# Frame 6: OOB summary
max_oob = oob_count.max() if oob_count.max() > 0 else 1
summary_colors = []
for i in range(n):
    if oob_count[i] == 0:
        summary_colors.append(COLORS["positive"])
    else:
        # Darker gray for more OOB
        frac = oob_count[i] / max_oob
        r = int(107 + (200 - 107) * (1 - frac))  # lighter for low count
        g = int(114 + (200 - 114) * (1 - frac))
        b_val = int(128 + (200 - 128) * (1 - frac))
        summary_colors.append(f"rgb({r},{g},{b_val})")

frames.append(go.Frame(
    data=[go.Bar(
        x=labels, y=values,
        marker=dict(color=summary_colors,
                    line=dict(width=1, color="white")),
        text=[f"OOB:{oob_count[i]}" for i in range(n)],
        textposition="outside",
    )],
    name="Summary",
    layout=go.Layout(
        title="OOB Summary — Darker = More Often Out-of-Bag"
    ),
))

# ── Initial figure ──────────────────────────────────────────────
fig = go.Figure(
    data=frames[0].data,
    frames=frames,
)

# ── Slider + play button ────────────────────────────────────────
slider_steps = []
frame_names = ["Original"] + [f"Sample {b+1}" for b in range(n_bootstrap)] + ["Summary"]
for name in frame_names:
    slider_steps.append(dict(
        method="animate",
        args=[[name], dict(mode="immediate",
                           frame=dict(duration=600, redraw=True),
                           transition=dict(duration=300))],
        label=name,
    ))

fig.update_layout(
    title="Bootstrap Sampling — Original Data",
    xaxis=dict(title="Observation"),
    yaxis=dict(title="Value", range=[values.min() - 3, values.max() + 4]),
    width=950, height=550,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.02, xanchor="left",
        y=1.15, yanchor="top",
        buttons=[
            dict(label="Play",
                 method="animate",
                 args=[None, dict(
                     frame=dict(duration=1000, redraw=True),
                     fromcurrent=True,
                     transition=dict(duration=400),
                 )]),
            dict(label="Pause",
                 method="animate",
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode="immediate",
                 )]),
        ],
    )],
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix="Frame: "),
        pad=dict(t=60),
        steps=slider_steps,
    )],
)

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bootstrap_sampling.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved to {out}")
    fig.show()
