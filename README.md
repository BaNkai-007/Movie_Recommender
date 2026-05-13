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
---

## 🔧 Try Different Movies

| Movie | Expected Cluster |
|---|---|
| Toy Story | Animation / Family |
| The Matrix | Sci-Fi & Adventure |
| The Notebook | Romance & Drama |
| Pulp Fiction | Dark Thriller / Indie |
| Shrek | Comedy / Animation |
