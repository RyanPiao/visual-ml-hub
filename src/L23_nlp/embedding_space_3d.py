"""
Document Embedding Space (3D)
==============================
100 synthetic FOMC-style document embeddings in 50-D, projected to 3D via
PCA (or UMAP if available).  Four categories: Hawkish, Dovish, Neutral,
Uncertain.

Slider: filter by decade (2000s / 2010s / 2020s / All).
Dropdown: colour by category, year, or uncertainty_score.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

np.random.seed(42)

# ------------------------------------------------------------------
# Synthetic data
# ------------------------------------------------------------------
N = 100
DIM = 50
CATEGORIES = ["Hawkish", "Dovish", "Neutral", "Uncertain"]
CAT_COLORS = [COLORS["negative"], COLORS["positive"],
              COLORS["secondary"], COLORS["gray"]]
CAT_MAP = {c: i for i, c in enumerate(CATEGORIES)}

# Cluster centres in 50-D
rng = np.random.RandomState(42)
centres = rng.randn(4, DIM) * 3.0
# Push apart
centres[0, :5] += 4
centres[1, :5] -= 4
centres[2, 5:10] += 4
centres[3, 5:10] -= 4

cat_idx = rng.choice(4, N, p=[0.28, 0.28, 0.24, 0.20])
embeddings = np.array([centres[c] + rng.randn(DIM) * 1.5 for c in cat_idx])

cat_labels = np.array([CATEGORIES[c] for c in cat_idx])

# Simulated metadata
years = rng.choice(np.arange(2000, 2025), N)
uncertainty = np.clip(0.3 + 0.5 * (cat_idx == 3).astype(float) + rng.randn(N) * 0.15, 0, 1)
importance = np.clip(rng.exponential(0.5, N) + 0.2, 0.3, 2.0)

# Short simulated snippets
SNIPPETS = {
    "Hawkish": [
        "Committee sees inflation risks...",
        "Tightening may be warranted...",
        "Labor market remains strong...",
        "Price pressures persist...",
    ],
    "Dovish": [
        "Accommodation supports recovery...",
        "Downside risks to growth...",
        "Easing financial conditions...",
        "Unemployment concerns linger...",
    ],
    "Neutral": [
        "Balanced assessment of risks...",
        "Data dependent approach...",
        "Conditions broadly stable...",
        "Monitoring developments...",
    ],
    "Uncertain": [
        "Outlook highly uncertain...",
        "Elevated uncertainty persists...",
        "Risks difficult to assess...",
        "Range of possible outcomes...",
    ],
}
snippets = np.array([SNIPPETS[cat_labels[i]][rng.randint(4)] for i in range(N)])

# ------------------------------------------------------------------
# Dimensionality reduction
# ------------------------------------------------------------------
try:
    from umap import UMAP
    reducer = UMAP(n_components=3, random_state=42, n_neighbors=15)
    X3 = reducer.fit_transform(embeddings)
    method_label = "UMAP"
except ImportError:
    reducer = PCA(n_components=3, random_state=42)
    X3 = reducer.fit_transform(embeddings)
    method_label = "PCA"

# ------------------------------------------------------------------
# Decade buckets
# ------------------------------------------------------------------
decade_label = np.where(years < 2010, "2000s",
               np.where(years < 2020, "2010s", "2020s"))
DECADES = ["All", "2000s", "2010s", "2020s"]


def color_array(mode):
    """Return (color_values, colorbar_title, is_numeric)."""
    if mode == "category":
        return np.array([CAT_COLORS[CAT_MAP[c]] for c in cat_labels]), None, False
    elif mode == "year":
        return years.astype(float), "Year", True
    else:  # uncertainty
        return uncertainty, "Uncertainty", True


# ------------------------------------------------------------------
# Build figure: 4 decades x 3 colour modes = 12 trace groups
# ------------------------------------------------------------------
COLOR_MODES = ["category", "year", "uncertainty_score"]
fig = go.Figure()

trace_groups = []  # list of (decade, color_mode, trace_indices)

for dec in DECADES:
    for cm in COLOR_MODES:
        mask = np.ones(N, dtype=bool) if dec == "All" else (decade_label == dec)
        cols, cbar_title, is_numeric = color_array(cm)

        idx_start = len(fig.data)

        if is_numeric:
            fig.add_trace(go.Scatter3d(
                x=X3[mask, 0], y=X3[mask, 1], z=X3[mask, 2],
                mode="markers",
                marker=dict(
                    size=importance[mask] * 5 + 3,
                    color=cols[mask],
                    colorscale="Viridis",
                    colorbar=dict(title=cbar_title, len=0.5),
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                text=snippets[mask],
                customdata=np.column_stack([
                    cat_labels[mask], years[mask].astype(str),
                    np.round(uncertainty[mask], 2).astype(str),
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "%{text}<br>"
                    "Uncertainty: %{customdata[2]}<extra></extra>"
                ),
                name=f"{dec} / {cm}",
                visible=False,
            ))
        else:
            # One trace per category for legend
            for ci, cat in enumerate(CATEGORIES):
                cat_mask = mask & (cat_idx == ci)
                if cat_mask.sum() == 0:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=X3[cat_mask, 0], y=X3[cat_mask, 1], z=X3[cat_mask, 2],
                    mode="markers",
                    marker=dict(
                        size=importance[cat_mask] * 5 + 3,
                        color=CAT_COLORS[ci],
                        opacity=0.85,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=snippets[cat_mask],
                    customdata=np.column_stack([
                        cat_labels[cat_mask], years[cat_mask].astype(str),
                        np.round(uncertainty[cat_mask], 2).astype(str),
                    ]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        "%{text}<br>"
                        "Uncertainty: %{customdata[2]}<extra></extra>"
                    ),
                    name=cat,
                    visible=False,
                ))

        idx_end = len(fig.data)
        trace_groups.append((dec, cm, list(range(idx_start, idx_end))))

# Default visible: All / category
for idx in trace_groups[0][2]:
    fig.data[idx].visible = True

# ------------------------------------------------------------------
# Slider for decade, dropdown for colour mode
# ------------------------------------------------------------------
def make_visibility(target_dec, target_cm):
    vis = [False] * len(fig.data)
    for dec, cm, indices in trace_groups:
        if dec == target_dec and cm == target_cm:
            for i in indices:
                vis[i] = True
    return vis


slider_steps = []
for di, dec in enumerate(DECADES):
    slider_steps.append(dict(
        method="update",
        args=[{"visible": make_visibility(dec, "category")}],
        label=dec,
    ))

dropdown_buttons = []
for cm in COLOR_MODES:
    dropdown_buttons.append(dict(
        method="update",
        args=[{"visible": make_visibility("All", cm)}],
        label=cm.replace("_", " ").title(),
    ))

layout = make_3d_layout(
    title=f"Document Embeddings: FOMC Minutes in 3D Space ({method_label})",
    x_title="Dim 1", y_title="Dim 2", z_title="Dim 3",
    width=950, height=720,
)
layout.update(
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix="Decade: "),
        pad=dict(t=60),
        steps=slider_steps,
    )],
    updatemenus=[dict(
        type="dropdown",
        buttons=dropdown_buttons,
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="white",
    )],
    legend=dict(orientation="v", x=1.02, y=0.95),
)
fig.update_layout(layout)

# ------------------------------------------------------------------
if __name__ == "__main__":
    fig.show()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "embedding_space_3d.html")
    fig.write_html(out)
    print(f"Saved → {out}")
