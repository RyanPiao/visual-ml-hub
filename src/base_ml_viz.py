"""
Shared colors, Plotly theme, and helper functions for ML visualizations.

All templates import from this module:
    from base_ml_viz import COLORS, ml_template, make_3d_layout
"""

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# ============================================================
# COLOR CONSTANTS (matching shared design system)
# ============================================================
COLORS = {
    "primary": "#1357c9",
    "positive": "#10b981",     # green — no-default, correct
    "negative": "#ef4444",     # red — default, error
    "secondary": "#3b82f6",    # blue
    "highlight": "#f97316",    # orange
    "purple": "#8b5cf6",
    "gray": "#6b7280",
    "light_blue": "#93c5fd",
    "light_red": "#fca5a5",
    "gold": "#eab308",         # ground truth
    "surface_bg": "#f9f9f9",
    "text": "#1a1c1c",
}

# Discrete class colors (for scatter, bar, etc.)
CLASS_COLORS = [COLORS["positive"], COLORS["negative"], COLORS["secondary"],
                COLORS["highlight"], COLORS["purple"]]

# Diverging colorscale: green (low) → white → red (high)
DIVERGING_SCALE = [
    [0.0, COLORS["positive"]],
    [0.5, "#ffffff"],
    [1.0, COLORS["negative"]],
]

# Sequential colorscale: light blue → dark blue
SEQUENTIAL_SCALE = [
    [0.0, "#e0f2fe"],
    [0.5, COLORS["secondary"]],
    [1.0, "#1e3a5f"],
]

# ============================================================
# PLOTLY TEMPLATE
# ============================================================
ml_template = go.layout.Template()
ml_template.layout = go.Layout(
    font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
    paper_bgcolor="white",
    plot_bgcolor=COLORS["surface_bg"],
    title=dict(font=dict(size=18, color=COLORS["text"]), x=0.5, xanchor="center"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.8)",
    ),
    margin=dict(l=60, r=30, t=80, b=60),
    xaxis=dict(gridcolor="#e2e2e2", zerolinecolor="#c4c7c7"),
    yaxis=dict(gridcolor="#e2e2e2", zerolinecolor="#c4c7c7"),
)
pio.templates["ml_viz"] = ml_template
pio.templates.default = "ml_viz"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_3d_layout(title="", x_title="", y_title="", z_title="",
                   width=800, height=650):
    """Standard 3D layout for ML visualizations."""
    return go.Layout(
        title=dict(text=title, font=dict(size=18)),
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(title=x_title, backgroundcolor=COLORS["surface_bg"],
                       gridcolor="#e2e2e2", showbackground=True),
            yaxis=dict(title=y_title, backgroundcolor=COLORS["surface_bg"],
                       gridcolor="#e2e2e2", showbackground=True),
            zaxis=dict(title=z_title, backgroundcolor=COLORS["surface_bg"],
                       gridcolor="#e2e2e2", showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )


def make_split_plane_mesh(axis, value, bounds, color, opacity=0.15):
    """Create a Mesh3d trace for a decision tree split plane.

    Args:
        axis: "x", "y", or "z" — which axis to split on
        value: the split threshold (in data coordinates)
        bounds: dict with keys x_min, x_max, y_min, y_max, z_min, z_max
        color: hex color string
        opacity: fill transparency (0-1)

    Returns:
        go.Mesh3d trace
    """
    xmin, xmax = bounds["x_min"], bounds["x_max"]
    ymin, ymax = bounds["y_min"], bounds["y_max"]
    zmin, zmax = bounds["z_min"], bounds["z_max"]

    if axis == "x":
        x = [value] * 4
        y = [ymin, ymax, ymax, ymin]
        z = [zmin, zmin, zmax, zmax]
    elif axis == "y":
        x = [xmin, xmax, xmax, xmin]
        y = [value] * 4
        z = [zmin, zmin, zmax, zmax]
    else:
        x = [xmin, xmax, xmax, xmin]
        y = [ymin, ymin, ymax, ymax]
        z = [value] * 4

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color,
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


def build_simple_tree(X, y, max_depth=3, min_samples=5, seed=42):
    """Build a simple decision tree and return split info for visualization.

    Returns list of splits: [{"axis": 0|1|2, "value": float,
                              "bounds": {...}, "depth": int}, ...]
    """
    rng = np.random.RandomState(seed)
    splits = []

    def _split(mask, bounds, depth):
        if depth >= max_depth or mask.sum() < min_samples:
            return

        best_gain = -1
        best_axis, best_val = 0, 0

        for ax in range(X.shape[1]):
            vals = np.unique(X[mask, ax])
            if len(vals) < 2:
                continue
            # Try a few candidate splits
            candidates = np.percentile(X[mask, ax], [25, 50, 75])
            for v in candidates:
                left = mask & (X[:, ax] <= v)
                right = mask & (X[:, ax] > v)
                if left.sum() < 2 or right.sum() < 2:
                    continue
                # Gini gain
                p_l = y[left].mean()
                p_r = y[right].mean()
                g_parent = 2 * y[mask].mean() * (1 - y[mask].mean())
                g_left = 2 * p_l * (1 - p_l)
                g_right = 2 * p_r * (1 - p_r)
                n = mask.sum()
                gain = g_parent - (left.sum() / n * g_left + right.sum() / n * g_right)
                if gain > best_gain:
                    best_gain = gain
                    best_axis = ax
                    best_val = v

        if best_gain <= 0:
            return

        axis_names = ["x", "y", "z"]
        splits.append({
            "axis": best_axis,
            "axis_name": axis_names[best_axis] if best_axis < 3 else f"x{best_axis}",
            "value": best_val,
            "bounds": dict(bounds),
            "depth": depth,
            "gini_gain": best_gain,
        })

        left_mask = mask & (X[:, best_axis] <= best_val)
        right_mask = mask & (X[:, best_axis] > best_val)

        keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        left_bounds = dict(bounds)
        left_bounds[keys[best_axis * 2 + 1]] = best_val
        right_bounds = dict(bounds)
        right_bounds[keys[best_axis * 2]] = best_val

        _split(left_mask, left_bounds, depth + 1)
        _split(right_mask, right_bounds, depth + 1)

    init_bounds = {
        "x_min": X[:, 0].min(), "x_max": X[:, 0].max(),
        "y_min": X[:, 1].min() if X.shape[1] > 1 else 0,
        "y_max": X[:, 1].max() if X.shape[1] > 1 else 1,
        "z_min": X[:, 2].min() if X.shape[1] > 2 else 0,
        "z_max": X[:, 2].max() if X.shape[1] > 2 else 1,
    }
    _split(np.ones(len(X), dtype=bool), init_bounds, 0)
    return splits
