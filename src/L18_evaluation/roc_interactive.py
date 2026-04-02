"""
Interactive ROC Curve
ROC curve with a slider that moves a dot along the curve.
Annotations show threshold, TPR, FPR, precision, accuracy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from base_ml_viz import COLORS, make_3d_layout

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# ── Synthetic binary data ───────────────────────────────────────
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, random_state=42, flip_y=0.1,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)[:, 1]

# ── Compute ROC at 50 thresholds ───────────────────────────────
thresholds = np.linspace(0.01, 0.99, 50)
fprs, tprs, precisions, accuracies = [], [], [], []

for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    fprs.append(fpr)
    tprs.append(tpr)
    precisions.append(prec)
    accuracies.append(acc)

fprs = np.array(fprs)
tprs = np.array(tprs)
precisions = np.array(precisions)
accuracies = np.array(accuracies)

# AUC approximation
sorted_idx = np.argsort(fprs)
auc_val = np.trapezoid(tprs[sorted_idx], fprs[sorted_idx])

# ── Build figure ────────────────────────────────────────────────
fig = go.Figure()

# Diagonal reference
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(color=COLORS["gray"], dash="dash", width=1),
    name="Random", showlegend=True,
))

# ROC curve
fig.add_trace(go.Scatter(
    x=fprs, y=tprs, mode="lines",
    line=dict(color=COLORS["primary"], width=3),
    name=f"ROC (AUC={auc_val:.3f})",
    fill="tozeroy",
    fillcolor="rgba(19,87,201,0.08)",
))

# Moving dot — initial position (threshold index 25 = midpoint)
init_idx = 25
fig.add_trace(go.Scatter(
    x=[fprs[init_idx]], y=[tprs[init_idx]],
    mode="markers",
    marker=dict(color=COLORS["negative"], size=14,
                line=dict(width=2, color="white")),
    name="Operating Point",
))

# ── Slider steps ────────────────────────────────────────────────
steps = []
for k in range(len(thresholds)):
    annotation_text = (
        f"<b>Threshold:</b> {thresholds[k]:.2f}<br>"
        f"<b>TPR (Recall):</b> {tprs[k]:.3f}<br>"
        f"<b>FPR:</b> {fprs[k]:.3f}<br>"
        f"<b>Precision:</b> {precisions[k]:.3f}<br>"
        f"<b>Accuracy:</b> {accuracies[k]:.3f}"
    )
    steps.append(dict(
        method="update",
        args=[
            {"x": [[0, 1], fprs.tolist(), [fprs[k]]],
             "y": [[0, 1], tprs.tolist(), [tprs[k]]]},
            {"annotations": [dict(
                x=0.98, y=0.05, xref="paper", yref="paper",
                text=annotation_text,
                showarrow=False,
                font=dict(size=13),
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=COLORS["gray"],
                borderwidth=1,
                borderpad=8,
            )]}
        ],
        label=f"{thresholds[k]:.2f}",
    ))

fig.update_layout(
    title="Interactive ROC Curve — Random Forest",
    xaxis=dict(title="False Positive Rate", range=[-0.02, 1.02]),
    yaxis=dict(title="True Positive Rate", range=[-0.02, 1.02],
               scaleanchor="x"),
    width=800, height=700,
    annotations=[dict(
        x=0.98, y=0.05, xref="paper", yref="paper",
        text=(f"<b>Threshold:</b> {thresholds[init_idx]:.2f}<br>"
              f"<b>TPR (Recall):</b> {tprs[init_idx]:.3f}<br>"
              f"<b>FPR:</b> {fprs[init_idx]:.3f}<br>"
              f"<b>Precision:</b> {precisions[init_idx]:.3f}<br>"
              f"<b>Accuracy:</b> {accuracies[init_idx]:.3f}"),
        showarrow=False,
        font=dict(size=13),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["gray"],
        borderwidth=1,
        borderpad=8,
    )],
    sliders=[dict(
        active=init_idx,
        currentvalue=dict(prefix="Threshold: "),
        pad=dict(t=60),
        steps=steps,
    )],
)

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "roc_interactive.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved to {out}")
    fig.show()
