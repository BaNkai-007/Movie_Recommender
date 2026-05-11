# 🎬 CineCluster — Movie Recommender (Unsupervised ML)

A college-level unsupervised machine learning project using **K-Means Clustering**
to group movies and power a recommendation engine, with a full interactive UI.

---

## 🧠 ML Concepts Used

| Concept | Where |
|---|---|
| **K-Means Clustering** | Groups similar movies into 8 clusters |
| **PCA (Dimensionality Reduction)** | Reduces features to 2D for visualization |
| **Feature Engineering** | Genre encoding, rating aggregation, year normalization |
| **MultiLabel Binarization** | One-hot encoding of multi-genre tags |
| **Silhouette Score** | Evaluates cluster quality |
| **Elbow Method** | Helps choose optimal K |

---

## 📁 Project Structure

```
movie-recommender/
├── app.py              ← Streamlit UI (main app)
├── model.py            ← ML pipeline (K-Means, PCA, features)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## ⚡ Setup & Run (5 minutes)

### Step 1 — Create a virtual environment
```bash
py -3.11 -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 🎯 How to Demo in Class

1. **Tab 1 — Recommend**: Type any movie (e.g. "Toy Story", "Matrix", "Batman")
   → Shows which cluster it belongs to + 6 recommendations from the same cluster

2. **Tab 2 — Cluster Map**: 2D PCA scatter plot — every dot is a movie,
   colored by cluster. Hover to see movie names.

3. **Tab 3 — Analysis**:
   - Cluster sizes bar chart
   - Avg rating per cluster
   - **Elbow + Silhouette curve** (explains why K=8)
   - Genre heatmap (proves clustering is meaningful)

---

## 📊 Talking Points for Presentation

- "We used **unsupervised** learning — no labels, the algorithm discovered structure on its own"
- "K-Means minimizes intra-cluster variance — movies in the same cluster are more similar to each other than to movies outside"
- "We validated K=8 using the **silhouette score** (closer to 1.0 = better separation)"
- "PCA helps us visualize 20+ dimensional data in 2D while preserving most variance"
- "The recommendation engine runs in O(n) — just find the cluster, rank by rating × log(popularity)"

---

## 🔧 Try Different Movies

| Movie | Expected Cluster |
|---|---|
| Toy Story | Animation / Family |
| The Matrix | Sci-Fi & Adventure |
| The Notebook | Romance & Drama |
| Pulp Fiction | Dark Thriller / Indie |
| Shrek | Comedy / Animation |
