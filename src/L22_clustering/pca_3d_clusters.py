"""
3D PCA Country Clusters
========================
80 synthetic countries with 6 economic indicators grouped into 4 income
tiers.  PCA projects to 3D; K-Means clusters are shown with a slider
for K = 2..8.  Misclassified points (cluster != true group) use an X marker.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ------------------------------------------------------------------
# Synthetic country data
# ------------------------------------------------------------------
N_PER_GROUP = 20
GROUP_NAMES = ["High-income", "Upper-middle", "Lower-middle", "Low-income"]
GROUP_COLORS = [COLORS["secondary"], COLORS["positive"],
                COLORS["highlight"], COLORS["negative"]]

# Centres: [GDP_pc, LifeExp, EduIdx, TradeOpen, Inflation, Gini]
CENTRES = np.array([
    [55, 80, 0.90, 70, 2.0, 30],   # High-income
    [25, 73, 0.72, 55, 5.0, 38],   # Upper-middle
    [10, 66, 0.55, 40, 8.0, 42],   # Lower-middle
    [ 3, 58, 0.35, 30, 12., 48],   # Low-income
], dtype=float)

SPREAD = np.array([12, 3, 0.08, 12, 2.0, 5])

X_raw = np.vstack([
    np.random.randn(N_PER_GROUP, 6) * SPREAD + CENTRES[g]
    for g in range(4)
])
true_labels = np.repeat(np.arange(4), N_PER_GROUP)

# Fictional country names
PREFIXES = ["Nova", "Alto", "Sol", "Mar", "Rio", "Val", "San", "Del",
            "Lor", "Tera", "Isla", "Bel", "Est", "Cor", "Mal", "Ven",
            "Pal", "Dor", "Cam", "Fon"]
SUFFIXES = ["nia", "stan", "land", "via", "burg", "dor", "ica", "lia",
            "ura", "eira", "ola", "tia", "ina", "ada", "osa", "ria",
            "ena", "bia", "dia", "fia"]
rng = np.random.RandomState(42)
names = [f"{PREFIXES[i % 20]}{SUFFIXES[(i * 3 + 7) % 20]}" for i in range(80)]

# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100

# ------------------------------------------------------------------
# Build traces for each K
# ------------------------------------------------------------------
K_RANGE = list(range(2, 9))  # 2..8

fig = go.Figure()

for ki, k in enumerate(K_RANGE):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    cluster_labels = km.labels_

    # Best-effort mapping of clusters to true groups for mismatch detection
    from scipy.stats import mode as _mode
    mapping = {}
    for c in range(k):
        mask_c = cluster_labels == c
        if mask_c.sum() > 0:
            mapping[c] = int(_mode(true_labels[mask_c], keepdims=False).mode)

    matched = np.array([mapping.get(c, -1) == tl
                        for c, tl in zip(cluster_labels, true_labels)])

    for g in range(4):
        for is_match in [True, False]:
            mask = (true_labels == g) & (matched == is_match)
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                z=X_pca[mask, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=GROUP_COLORS[g],
                    symbol="circle" if is_match else "x",
                    opacity=0.85,
                    line=dict(width=0.5, color="white")),
                name=f"{GROUP_NAMES[g]} ({'correct' if is_match else 'misclass.'})",
                visible=(ki == 0),
                text=[names[j] for j in np.where(mask)[0]],
                customdata=np.column_stack([
                    [GROUP_NAMES[g]] * mask.sum(),
                    [f"Cluster {cluster_labels[j]}" for j in np.where(mask)[0]],
                ]),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "True group: %{customdata[0]}<br>"
                    "Assigned: %{customdata[1]}<br>"
                    "PC1: %{x:.2f}  PC2: %{y:.2f}  PC3: %{z:.2f}"
                    "<extra></extra>"
                )))

# Count traces per K step
traces_per_k = []
idx = 0
for ki, k in enumerate(K_RANGE):
    cnt = 0
    # re-run to count (cheap)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    cl = km.labels_
    mapping = {}
    for c in range(k):
        mc = cl == c
        if mc.sum() > 0:
            mapping[c] = int(_mode(true_labels[mc], keepdims=False).mode)
    matched = np.array([mapping.get(c, -1) == tl for c, tl in zip(cl, true_labels)])
    for g in range(4):
        for is_match in [True, False]:
            if ((true_labels == g) & (matched == is_match)).sum() > 0:
                cnt += 1
    traces_per_k.append(cnt)

# Slider
steps = []
offset = 0
for ki, k in enumerate(K_RANGE):
    vis = [False] * len(fig.data)
    for j in range(traces_per_k[ki]):
        vis[offset + j] = True
    steps.append(dict(
        method="update",
        args=[{"visible": vis}],
        label=str(k)))
    offset += traces_per_k[ki]

layout = make_3d_layout(
    title="Clustering Economies: PCA Projection with K-Means",
    x_title=f"PC1 ({var_explained[0]:.1f}% var)",
    y_title=f"PC2 ({var_explained[1]:.1f}% var)",
    z_title=f"PC3 ({var_explained[2]:.1f}% var)",
    width=950,
    height=720)
layout.update(
    title=dict(text=(
        "<b>Clustering Economies: PCA Projection with K-Means</b><br>"
        "<span style='font-size:13px; color:#6b7280'>"
        f"3 principal components explain {sum(var_explained):.0f}% of total variance</span>"
    ), font=dict(size=16), x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=60, b=80),
        sliders=[dict(
        active=0,
        currentvalue=dict(prefix="K = "),
        pad=dict(t=40),
        steps=steps)],
    legend=dict(orientation="v", x=1.02, y=0.95))
fig.update_layout(layout)

# ------------------------------------------------------------------
if __name__ == "__main__":
    fig.show()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "pca_3d_clusters.html")
    fig.write_html(out)
    print(f"Saved → {out}")
