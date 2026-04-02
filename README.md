# Visual ML Hub

Interactive ML & statistics visualizations for **Econ 5200 / 3916** — Lectures 17 through 24.

Students rotate, zoom, hover, and drag sliders. Zero installation — just click and explore.

## Live Site

**[https://RyanPiao.github.io/visual-ml-hub/](https://RyanPiao.github.io/visual-ml-hub/)**

## What's Inside

| Lecture | Topic | Visualizations |
|---------|-------|---------------|
| **L17** | Classification | 3D logistic sigmoid surface, 2D decision boundary |
| **L18** | Model Evaluation | Cost-loss landscape (3D), ROC with threshold slider |
| **L19** | Trees & RF | 3D splits, forest averaging, SHAP interaction, SHAP waterfall, bootstrap, gradient boosting |
| **L20** | Time Series | 3D decomposition stack (trend + seasonal + residual) |
| **L22** | Clustering | 3D PCA country clusters with K slider |
| **L23** | NLP | FOMC document embedding space (3D) |
| **L24** | Causal ML | CATE heterogeneity surface (5200) |

**14 interactive charts total.**

## Canvas LMS Embedding

```html
<iframe src="https://RyanPiao.github.io/visual-ml-hub/tree_splits_3d.html"
        width="100%" height="700" frameborder="0"></iframe>
```

Replace the filename with any chart from the hub.

## Regenerating Charts

```bash
# Activate Python environment
source ~/.venv/bin/activate  # or your venv path
pip install plotly scikit-learn shap numpy pandas

# Run any template to regenerate its HTML
cd src/L19_trees
python tree_splits_3d.py     # generates .html in same folder

# Copy to docs/ for GitHub Pages
cp src/L19_trees/tree_splits_3d.html docs/
git add docs/ && git commit -m "Update tree splits" && git push
```

## Tech Stack

- **Plotly** — 3D scatter, surfaces, sliders, animation
- **scikit-learn** — ML models (trees, RF, logistic, KMeans)
- **SHAP** — Feature attribution
- **GitHub Pages** — Free hosting from `docs/` folder
