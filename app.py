"""
🎬 CineCluster — Movie Recommender using K-Means Unsupervised Learning
Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from model import (
    run_pipeline, recommend,
    find_optimal_k, CLUSTER_NAMES, CLUSTER_DESCRIPTIONS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineCluster",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c28;
    --accent: #e8b04b;
    --accent2: #e85d4b;
    --text: #e8e8f0;
    --muted: #7a7a9a;
    --border: rgba(232, 176, 75, 0.15);
}

html, body, [class*="css"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 6px;
    color: var(--accent);
    margin: 0;
    line-height: 1;
    text-shadow: 0 0 60px rgba(232, 176, 75, 0.3);
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* Cards */
.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: var(--accent); }
.rec-title { font-weight: 600; font-size: 1rem; color: var(--text); }
.rec-meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }
.rec-badge {
    display: inline-block;
    background: rgba(232, 176, 75, 0.12);
    border: 1px solid rgba(232, 176, 75, 0.3);
    color: var(--accent);
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 0.4rem;
}

/* Cluster pill */
.cluster-pill {
    background: linear-gradient(135deg, #e8b04b22, #e85d4b22);
    border: 1px solid var(--accent);
    border-radius: 30px;
    padding: 0.6rem 1.4rem;
    font-size: 1.1rem;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}

/* Stats row */
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-num { font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; color: var(--accent); }
.stat-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Input */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

div[data-testid="stSelectbox"] > div {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 1px;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return run_pipeline()


with st.spinner("🎬 Training K-Means clusters on movie data..."):
    movies_df, features, km_model, labels, coords, var_ratio, genre_cols = load_model()

n_movies = len(movies_df)
n_clusters = km_model.n_clusters

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineCluster")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. Movies are encoded by **genre**, **rating**, **popularity**, and **year**
2. **K-Means** groups them into 8 clusters of similar films
3. **PCA** reduces features to 2D for visualization
4. Recommendations come from the **same cluster**
    """)
    st.markdown("---")

    # Model stats
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(features, labels, sample_size=min(500, n_movies))
    st.markdown("**Model Metrics**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clusters", n_clusters)
        st.metric("Movies", f"{n_movies:,}")
    with col2:
        st.metric("Silhouette", f"{sil:.3f}")
        st.metric("PCA Var", f"{sum(var_ratio)*100:.1f}%")

    st.markdown("---")
    st.markdown("**Project Info**")
    st.caption("Unsupervised ML · K-Means · PCA\nDataset: MovieLens")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>CINECLUSTER</h1>
    <p>Unsupervised Movie Recommendation Engine · K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TOP STATS ROW
# ─────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f"""<div class="stat-box"><div class="stat-num">{n_movies:,}</div><div class="stat-label">Movies Clustered</div></div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""<div class="stat-box"><div class="stat-num">{n_clusters}</div><div class="stat-label">K-Means Clusters</div></div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class="stat-box"><div class="stat-num">{sil:.2f}</div><div class="stat-label">Silhouette Score</div></div>""", unsafe_allow_html=True)
with s4:
    st.markdown(f"""<div class="stat-box"><div class="stat-num">{len(genre_cols)}</div><div class="stat-label">Genre Features</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Recommend", "🗺️ Cluster Map", "📊 Analysis"])


# ══════════════════════════════════════════════
# TAB 1 — RECOMMEND
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown("### Find Similar Movies")
        st.markdown("Type any movie title to discover its cluster and get recommendations.")

        # Movie search
        movie_input = st.text_input("🎬 Movie Title", placeholder="e.g. Toy Story, Batman, Matrix...")

        top_n = st.slider("Number of Recommendations", 3, 12, 6)

        search_btn = st.button("GET RECOMMENDATIONS")

        if search_btn and movie_input:
            with st.spinner("Searching cluster..."):
                movie, cluster_id, recs = recommend(
                    movie_input, movies_df, features, labels, top_n=top_n
                )

            if movie is None:
                st.error(f"❌ No movie found matching **'{movie_input}'**. Try a different title.")
            else:
                st.markdown("---")
                st.markdown("**Query Movie**")
                st.markdown(f"### {movie['title']}")

                cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
                st.markdown(f"""<div class="cluster-pill">{cluster_name}</div>""", unsafe_allow_html=True)
                st.caption(CLUSTER_DESCRIPTIONS.get(cluster_id, ""))

                # Show genre tags
                if isinstance(movie.get("genres_list"), list):
                    genres_str = " · ".join(movie["genres_list"])
                    st.caption(f"🏷️ {genres_str}")

                # Rating + year
                yr = int(movie["year"]) if not pd.isna(movie.get("year", float("nan"))) else "N/A"
                rt = f"{movie['avg_rating']:.1f}⭐" if movie.get("avg_rating") else "N/A"
                st.caption(f"📅 {yr}  ·  {rt}  ·  {int(movie.get('num_ratings', 0)):,} ratings")

                # Store for right panel
                st.session_state["recs"] = recs
                st.session_state["cluster_id"] = cluster_id
                st.session_state["searched"] = True

    with col_right:
        st.markdown("### Recommendations")
        if st.session_state.get("searched") and not st.session_state.get("recs", pd.DataFrame()).empty:
            recs = st.session_state["recs"]
            cluster_id = st.session_state["cluster_id"]
            cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")

            st.caption(f"Top picks from **{cluster_name}** cluster · sorted by rating × popularity")

            for i, (_, row) in enumerate(recs.iterrows()):
                yr = int(row["year"]) if not pd.isna(row.get("year", float("nan"))) else "N/A"
                rating = f"{row['avg_rating']:.1f}⭐" if row.get("avg_rating") else ""
                genres = " · ".join(row["genres_list"]) if isinstance(row.get("genres_list"), list) else ""
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">#{i+1} — {row['title']}</div>
                    <div class="rec-meta">{rating} &nbsp;·&nbsp; 📅 {yr} &nbsp;·&nbsp; {int(row.get('num_ratings',0)):,} ratings</div>
                    <span class="rec-badge">{genres[:60]}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#7a7a9a; padding:3rem; text-align:center; border:1px dashed rgba(232,176,75,0.15); border-radius:12px;">
                ← Enter a movie title and click<br><strong>GET RECOMMENDATIONS</strong>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — CLUSTER MAP
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ Movie Cluster Map (PCA 2D)")
    st.caption(f"Each dot = one movie. K-Means found **{n_clusters} natural groupings**. PCA explains {sum(var_ratio)*100:.1f}% of variance.")

    # Build plot df
    plot_df = movies_df.copy()
    plot_df["cluster_name"] = plot_df["cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))
    plot_df["hover"] = plot_df["title"] + "<br>" + plot_df["genres"] + "<br>⭐ " + plot_df["avg_rating"].round(2).astype(str)
    plot_df["size"] = np.log1p(plot_df["num_ratings"]) + 3

    colors = [
        "#e8b04b", "#e85d4b", "#4be8b0", "#4b8de8",
        "#b04be8", "#e84bb0", "#4be84b", "#e8e84b"
    ]

    fig = px.scatter(
        plot_df,
        x="pca_x", y="pca_y",
        color="cluster_name",
        size="size",
        hover_name="title",
        hover_data={"pca_x": False, "pca_y": False, "size": False, "genres": True, "avg_rating": ":.2f"},
        color_discrete_sequence=colors,
        title="",
        height=580,
    )
    fig.update_traces(marker=dict(opacity=0.72, line=dict(width=0)))
    fig.update_layout(
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        font_color="#e8e8f0",
        legend_title_text="Cluster",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(232,176,75,0.2)", borderwidth=1),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Analysis")

    col_a, col_b = st.columns(2)

    # Cluster sizes
    with col_a:
        cluster_counts = movies_df["cluster"].value_counts().sort_index()
        cluster_counts.index = cluster_counts.index.map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))

        fig2 = px.bar(
            x=cluster_counts.values,
            y=cluster_counts.index,
            orientation="h",
            title="Movies per Cluster",
            color=cluster_counts.values,
            color_continuous_scale=["#e85d4b", "#e8b04b"],
            height=360,
        )
        fig2.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#13131a",
            font_color="#e8e8f0", showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Movie Count", yaxis_title="",
            margin=dict(l=10, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Avg rating per cluster
    with col_b:
        avg_ratings = movies_df.groupby("cluster")["avg_rating"].mean().sort_index()
        avg_ratings.index = avg_ratings.index.map(lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}"))

        fig3 = px.bar(
            x=avg_ratings.values,
            y=avg_ratings.index,
            orientation="h",
            title="Avg Rating per Cluster",
            color=avg_ratings.values,
            color_continuous_scale=["#4b8de8", "#4be8b0"],
            height=360,
        )
        fig3.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#13131a",
            font_color="#e8e8f0", showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Avg Rating", yaxis_title="",
            margin=dict(l=10, r=20, t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Elbow + Silhouette
    st.markdown("### Optimal K — Elbow & Silhouette Analysis")
    st.caption("This shows WHY we picked K=8 clusters. The elbow method and silhouette score guide the choice.")

    with st.spinner("Computing elbow curve..."):
        k_vals, inertias, silhouettes = find_optimal_k(features, k_range=range(3, 13))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=k_vals, y=inertias, mode="lines+markers",
        name="Inertia (Elbow)", line=dict(color="#e8b04b", width=2),
        marker=dict(size=7)
    ))
    fig4.add_trace(go.Scatter(
        x=k_vals, y=[s * max(inertias) for s in silhouettes],
        mode="lines+markers",
        name="Silhouette × scale", line=dict(color="#4be8b0", width=2, dash="dot"),
        marker=dict(size=7)
    ))
    fig4.add_vline(x=8, line_dash="dash", line_color="#e85d4b", annotation_text="K=8 chosen")
    fig4.update_layout(
        plot_bgcolor="#0a0a0f", paper_bgcolor="#13131a",
        font_color="#e8e8f0", height=350,
        xaxis_title="Number of Clusters (K)", yaxis_title="Score",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=20, b=40)
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Genre heatmap
    st.markdown("### Genre Distribution Across Clusters")
    st.caption("Which genres dominate each cluster — proves the clustering is meaningful.")

    genre_cols_list = [c for c in features.columns if c in [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]]
    if genre_cols_list:
        heat_data = []
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                means = features.loc[movies_df[movies_df["cluster"] == c].index, genre_cols_list].mean()
                heat_data.append(means.values)

        heat_df = pd.DataFrame(heat_data, columns=genre_cols_list,
                               index=[CLUSTER_NAMES.get(i, f"C{i}") for i in range(n_clusters)])

        fig5 = px.imshow(
            heat_df,
            color_continuous_scale='YlOrBr',
            title="",
            aspect="auto",
            height=380,
        )
        fig5.update_layout(
            paper_bgcolor="#13131a", font_color="#e8e8f0",
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)
