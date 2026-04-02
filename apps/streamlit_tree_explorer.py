"""
Streamlit App: Interactive Decision Tree & Random Forest Explorer

Sidebar controls:
  - Dataset: synthetic loan default
  - Tree depth slider (1-15)
  - Number of trees B slider (1-200)
  - Max features (sqrt, log2, all)
  - Show/hide split planes, true surface

Main panel:
  - Tab 1: 3D scatter + split planes (decision tree)
  - Tab 2: 3D surface (forest averaging vs truth)
  - Tab 3: Metrics table (RMSE, R², accuracy, tree count)

Deploy:
    # Local
    streamlit run streamlit_tree_explorer.py

    # Streamlit Community Cloud
    Push to GitHub → connect at share.streamlit.io

    # Canvas embed
    <iframe src="https://your-app.streamlit.app" width="100%" height="800"></iframe>

Course:  Econ 5200 / 3916
Topic:   Lecture 19 — Tree-Based Models
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree & Random Forest Explorer",
    page_icon="🌲",
    layout="wide",
)

# ── Colors ───────────────────────────────────────────────────
COLORS = {
    "positive": "#10b981",
    "negative": "#ef4444",
    "secondary": "#3b82f6",
    "highlight": "#f97316",
    "primary": "#1357c9",
    "gold": "#eab308",
    "gray": "#6b7280",
}


# ── Data generation ──────────────────────────────────────────
@st.cache_data
def generate_classification_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    income = rng.uniform(25, 85, n)
    credit = rng.uniform(520, 780, n)
    loan = rng.uniform(5, 35, n)
    score = 0.06 * (income - 52) + 0.008 * (credit - 640) - 0.05 * (loan - 20)
    prob = 1 / (1 + np.exp(score))
    labels = (rng.random(n) < prob).astype(int)
    X = np.column_stack([income, credit, loan])
    return X, labels, ["Income ($K)", "Credit Score", "Loan ($K)"]


@st.cache_data
def generate_regression_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    y_true = 3 + 1.5 * np.sin(0.7 * x1) + np.cos(0.5 * x2) - 0.2 * x1 * x2 / 10
    y = y_true + rng.normal(0, 0.6, n)
    X = np.column_stack([x1, x2])
    return X, y, y_true, ["Feature 1", "Feature 2"]


# ── Split plane helper ───────────────────────────────────────
def get_split_planes(tree_model, X, feature_names):
    tree = tree_model.tree_
    planes = []

    def walk(node, bounds, depth):
        if tree.feature[node] < 0:
            return
        feat = tree.feature[node]
        thresh = tree.threshold[node]
        planes.append({"feat": feat, "thresh": thresh,
                       "bounds": dict(bounds), "depth": depth,
                       "name": feature_names[feat]})
        keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        lb = dict(bounds)
        lb[keys[feat * 2 + 1]] = thresh
        rb = dict(bounds)
        rb[keys[feat * 2]] = thresh
        walk(tree.children_left[node], lb, depth + 1)
        walk(tree.children_right[node], rb, depth + 1)

    b0 = {"x_min": X[:, 0].min() - 1, "x_max": X[:, 0].max() + 1,
           "y_min": X[:, 1].min() - 5, "y_max": X[:, 1].max() + 5,
           "z_min": X[:, 2].min() - 1 if X.shape[1] > 2 else 0,
           "z_max": X[:, 2].max() + 1 if X.shape[1] > 2 else 1}
    walk(0, b0, 0)
    return planes


def plane_mesh(feat, thresh, bounds, color, opacity=0.12):
    axis_map = {0: "x", 1: "y", 2: "z"}
    ax = axis_map[feat]
    xmin, xmax = bounds["x_min"], bounds["x_max"]
    ymin, ymax = bounds["y_min"], bounds["y_max"]
    zmin, zmax = bounds["z_min"], bounds["z_max"]
    if ax == "x":
        x, y, z = [thresh]*4, [ymin, ymax, ymax, ymin], [zmin, zmin, zmax, zmax]
    elif ax == "y":
        x, y, z = [xmin, xmax, xmax, xmin], [thresh]*4, [zmin, zmin, zmax, zmax]
    else:
        x, y, z = [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], [thresh]*4
    return go.Mesh3d(x=x, y=y, z=z, i=[0,0], j=[1,2], k=[2,3],
                     color=color, opacity=opacity, hoverinfo="skip", showlegend=False)


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🌲 Tree Explorer")
st.sidebar.markdown("---")

depth = st.sidebar.slider("Tree Depth", 1, 15, 4, help="Max depth of each decision tree")
n_trees = st.sidebar.slider("Number of Trees (B)", 1, 200, 1,
                             help="B=1 is a single tree. B>1 is a Random Forest.")
max_feat = st.sidebar.selectbox("Max Features per Split",
                                ["All", "sqrt", "log2"],
                                help="'sqrt' is the RF default for classification")
show_planes = st.sidebar.checkbox("Show split planes", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("- Drag the **depth slider** to see more/fewer splits")
st.sidebar.markdown("- Increase **B** to see ensemble smoothing")
st.sidebar.markdown("- **Rotate** the 3D plot by click-dragging")
st.sidebar.markdown("- **Hover** on points to see values")

# ── Main content ─────────────────────────────────────────────
st.title("Decision Tree & Random Forest Explorer")
st.caption("Lecture 19: Tree-Based Models — Econ 5200 / 3916")

tab1, tab2, tab3 = st.tabs(["🔹 Classification (3D Splits)", "🔹 Regression (Surface)", "📊 Metrics"])

# ── Tab 1: Classification ────────────────────────────────────
with tab1:
    X_cls, y_cls, feat_names = generate_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    mf = None if max_feat == "All" else max_feat.lower()

    if n_trees == 1:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        n_leaves = clf.get_n_leaves()
        model_name = f"Decision Tree (depth={depth})"
    else:
        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth,
                                      max_features=mf, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        n_leaves = sum(t.get_n_leaves() for t in clf.estimators_)
        model_name = f"Random Forest (B={n_trees}, depth={depth})"

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model_name)
    col2.metric("Test Accuracy", f"{acc:.1%}")
    col3.metric("Total Leaves", f"{n_leaves:,}")

    # 3D scatter
    fig = go.Figure()
    for label, name, color in [(0, "No Default", COLORS["positive"]),
                                (1, "Default", COLORS["negative"])]:
        mask = y_cls == label
        fig.add_trace(go.Scatter3d(
            x=X_cls[mask, 0], y=X_cls[mask, 1], z=X_cls[mask, 2],
            mode="markers", name=name,
            marker=dict(size=3, color=color, opacity=0.7),
            hovertemplate=f"Income: %{{x:.0f}}K<br>Credit: %{{y:.0f}}<br>Loan: $%{{z:.0f}}K<br><b>{name}</b><extra></extra>",
        ))

    # Split planes
    if show_planes and n_trees == 1:
        depth_colors = [COLORS["secondary"], COLORS["highlight"], COLORS["primary"],
                        COLORS["gray"], "#a78bfa", "#fbbf24", "#6ee7b7"]
        planes = get_split_planes(clf, X_cls, feat_names)
        for p in planes:
            c = depth_colors[min(p["depth"], len(depth_colors) - 1)]
            fig.add_trace(plane_mesh(p["feat"], p["thresh"], p["bounds"], c, 0.12))

    fig.update_layout(
        scene=dict(
            xaxis_title="Income ($K)", yaxis_title="Credit Score", zaxis_title="Loan ($K)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=600, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_trees > 1 and show_planes:
        st.info("Split planes are shown for single trees only (B=1). Set B=1 to see planes, then increase B to see the ensemble effect in the Regression tab.")

# ── Tab 2: Regression Surface ────────────────────────────────
with tab2:
    X_reg, y_reg, y_true_reg, feat_names_reg = generate_regression_data()

    mf_reg = None if max_feat == "All" else max_feat.lower()

    if n_trees == 1:
        reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    else:
        reg = RandomForestRegressor(n_estimators=n_trees, max_depth=depth,
                                     max_features=mf_reg, random_state=42, n_jobs=-1)
    reg.fit(X_reg, y_reg)

    # Surface grid
    g1 = np.linspace(0, 10, 40)
    g2 = np.linspace(0, 10, 40)
    G1, G2 = np.meshgrid(g1, g2)
    X_grid = np.column_stack([G1.ravel(), G2.ravel()])
    Z_pred = reg.predict(X_grid).reshape(40, 40)
    Z_true = np.array([[3 + 1.5*np.sin(0.7*a) + np.cos(0.5*b) - 0.2*a*b/10
                         for a, b in zip(r1, r2)] for r1, r2 in zip(G1, G2)])

    rmse = np.sqrt(mean_squared_error(Z_true.ravel(), Z_pred.ravel()))

    col1, col2 = st.columns(2)
    col1.metric("Surface RMSE", f"{rmse:.3f}")
    col2.metric("Trees", f"{n_trees}")

    fig2 = go.Figure()
    fig2.add_trace(go.Surface(x=G1, y=G2, z=Z_true, opacity=0.35, showscale=False,
                              colorscale=[[0, COLORS["gold"]], [1, COLORS["gold"]]],
                              name="True Function"))
    fig2.add_trace(go.Surface(x=G1, y=G2, z=Z_pred, opacity=0.7, showscale=False,
                              colorscale=[[0, "#dbeafe"], [1, COLORS["primary"]]],
                              name=f"Prediction (B={n_trees})"))
    fig2.update_layout(
        scene=dict(
            xaxis_title="Feature 1", yaxis_title="Feature 2", zaxis_title="Prediction",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        height=600, margin=dict(l=0, r=0, t=30, b=0),
        annotations=[dict(text="Gold = truth | Blue = model prediction",
                          x=0.5, y=0, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=12, color=COLORS["gray"]))],
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Try it:** Set B=1 and depth=8 → blocky staircase. Then slide B up to 100 → surface smooths toward gold truth.")

# ── Tab 3: Metrics ───────────────────────────────────────────
with tab3:
    st.subheader("Model Comparison")
    st.markdown("How does performance change with tree depth and ensemble size?")

    depths = list(range(1, 16))
    results = []
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    for d in depths:
        # Single tree
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train_r, y_train_r)
        pred_dt = dt.predict(X_test_r)
        results.append({"Depth": d, "Model": "Single Tree",
                        "RMSE": np.sqrt(mean_squared_error(y_test_r, pred_dt)),
                        "R²": r2_score(y_test_r, pred_dt)})
        # Random Forest (50 trees)
        if n_trees > 1:
            rf = RandomForestRegressor(n_estimators=min(n_trees, 50), max_depth=d,
                                        random_state=42, n_jobs=-1)
            rf.fit(X_train_r, y_train_r)
            pred_rf = rf.predict(X_test_r)
            results.append({"Depth": d, "Model": f"RF (B={min(n_trees, 50)})",
                            "RMSE": np.sqrt(mean_squared_error(y_test_r, pred_rf)),
                            "R²": r2_score(y_test_r, pred_rf)})

    df = pd.DataFrame(results)

    import plotly.express as px
    fig3 = px.line(df, x="Depth", y="RMSE", color="Model",
                   title="Test RMSE vs. Tree Depth",
                   color_discrete_sequence=[COLORS["negative"], COLORS["primary"]])
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(df, x="Depth", y="R²", color="Model",
                   title="Test R² vs. Tree Depth",
                   color_discrete_sequence=[COLORS["negative"], COLORS["primary"]])
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    **Key insight:** Single trees overfit at high depth (RMSE rises).
    Random Forests maintain low RMSE even at high depth because averaging
    reduces variance — this is the core benefit of ensembles.
    """)
