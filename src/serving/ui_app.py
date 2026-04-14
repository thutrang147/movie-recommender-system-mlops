

"""MVP Streamlit UI: Personalized movie recommendations powered by BPR."""
from __future__ import annotations


# --- Standard imports ---
import sys
from pathlib import Path
import pickle
from typing import Dict
import pandas as pd  # type: ignore[import-not-found]
import streamlit as st

# --- Ensure project root is in sys.path after all imports ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))



def get_bpr_recommendations(
    user_id: int,
    top_k: int,
    bpr_bundle: Dict,
):
    """Get recommendations from BPR model."""
    try:
        from src.models.bpr import recommend_with_bundle as recommend_with_bpr
        recommendations = recommend_with_bpr(
            bundle=bpr_bundle,
            user_id=user_id,
            top_k=top_k,
        )
        if not recommendations:
            return []
        return recommendations
    except Exception as e:
        st.error(f"BPR Model Error: {str(e)}")
        return []

@st.cache_resource
def load_bpr_model(model_path: Path) -> Dict:
    """Load BPR model bundle from pickle file."""
    if not model_path.exists():
        st.error(f"BPR model not found at {model_path}")
        return {}
    
    with open(model_path, "rb") as file:
        bundle = pickle.load(file)
    return bundle





@st.cache_resource
def load_user_history() -> pd.DataFrame:
    """Load user rating history for displaying watched movies."""
    project_root = get_project_root()
    # Load from processed data
    history_path = project_root / "data" / "processed" / "ratings_preprocessed.parquet"
    if history_path.exists():
        return pd.read_parquet(history_path)
    # Fallback to split data
    split_path = project_root / "data" / "split" / "train.parquet"
    if split_path.exists():
        return pd.read_parquet(split_path)
    return pd.DataFrame()


def get_user_watched_movies(user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Get movies watched by a user, sorted by rating."""
    if ratings_df.empty:
        return pd.DataFrame()
    
    user_ratings = ratings_df[ratings_df["user_id"] == user_id].sort_values("rating", ascending=False).head(top_n)
    if user_ratings.empty:
        return pd.DataFrame()
    
    watched_movies = []
    for _, row in user_ratings.iterrows():
        movie_info = get_movie_info(int(row["movie_id"]), movies_df)
        watched_movies.append({
            "Movie ID": movie_info["movie_id"],
            "Title": movie_info["title"],
            "User Rating": f"{row['rating']:.1f}/5.0",
        })
    
    return pd.DataFrame(watched_movies)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[2]


def load_movies_data() -> pd.DataFrame:
    """Load cleaned movies data."""
    project_root = get_project_root()
    movies_path = project_root / "data" / "interim" / "movies_cleaned.csv"
    if movies_path.exists():
        return pd.read_csv(movies_path)
    return pd.DataFrame()


def get_movie_info(movie_id: int, movies_df: pd.DataFrame) -> Dict:
    """Get movie information by ID."""
    movie = movies_df[movies_df["movie_id"] == movie_id]
    if movie.empty:
        return {"movie_id": movie_id, "title": "Unknown", "genres": "N/A"}
    
    movie_row = movie.iloc[0]
    return {
        "movie_id": int(movie_row.get("movie_id", movie_id)),
        "title": str(movie_row.get("title", "Unknown")),
        "genres": str(movie_row.get("genres", "N/A")),
    }


def format_recommendations(
    recommendations: list,
    scores: list | None,
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Format recommendations into a nice DataFrame for display."""
    rows = []
    for rank, rec in enumerate(recommendations, 1):
        if isinstance(rec, dict):
            movie_id = rec.get("movie_id", None)
            score = rec.get("score", None)
        else:
            movie_id = rec[0] if isinstance(rec, (tuple, list)) else rec
            score = scores[rank - 1] if scores and rank <= len(scores) else None
        movie_info = get_movie_info(movie_id, movies_df)
        # Ensure score is always float and formatted to 2 decimals if possible
        if score is not None:
            try:
                score_val = float(score)
                score_str = f"{score_val:.2f}"
            except Exception:
                score_str = str(score)
        else:
            score_str = "N/A"
        rows.append({
            "#": rank,
            "Title": movie_info["title"],
            "Score": score_str,
        })
    df = pd.DataFrame(rows)
    df = df[["#", "Title", "Score"]]
    return df

def show_model_info(bpr_bundle: Dict) -> None:
    """Display BPR model information and statistics."""
    with st.sidebar.expander("📋 Model Information"):
        st.markdown("### 🤖 BPR Model")
        st.markdown("**Theory:** Bayesian Personalized Ranking optimizes for ranking by learning to predict relative preferences between items for each user.")
        if bpr_bundle:
            config = bpr_bundle.get("config", {})
            st.write(f"**Algorithm:** {bpr_bundle.get('algorithm', 'Unknown')}")
            st.write(f"**Model Config:** {config}")
            for key, value in config.items():
                st.write(f"  - {key}: {value}")


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ===== LOAD DATA =====
    project_root = get_project_root()
    bpr_model_path = project_root / "models" / "personalized" / "bpr_model.pkl"
    with st.spinner("Loading model and data..."):
        bpr_bundle = load_bpr_model(bpr_model_path)
        movies_df = load_movies_data()
        ratings_df = load_user_history()
    if not bpr_bundle:
        st.error("❌ BPR model not loaded. Please check model path.")
        st.stop()

    # ===== SIDEBAR =====
    st.sidebar.header("⚙️ Settings")
    user_id = st.sidebar.number_input("User ID", min_value=1, value=1, step=1)
    top_k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=10)
    show_history = st.sidebar.checkbox("Show watched history", value=False)
    show_model_info = st.sidebar.checkbox("Show model information", value=False)

    # ===== USER SNAPSHOT =====
    user_history = get_user_watched_movies(user_id, ratings_df, movies_df, 100)
    watched_count = len(user_history)
    avg_rating = user_history["User Rating"].str.extract(r"([\d.]+)").astype(float).mean().values[0] if not user_history.empty else None
    st.markdown("<h2 style='margin-bottom:0.2em;'>🎬 Movie Recommender System</h2>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.1em; color:gray; margin-bottom:0.5em;'>Personalized movie recommendations powered by BPR</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex;gap:2em;margin-bottom:0.5em;'>"
        f"<div><b>User:</b> {user_id}</div>"
        f"<div><b>Watched:</b> {watched_count} movies</div>"
        f"<div><b>Avg rating:</b> {avg_rating:.1f}/5.0</div>"
        f"<div><b>Top N:</b> {top_k}</div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888;font-size:0.98em;margin-bottom:0.5em;'>Recommendations exclude movies already watched by the user.</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888;font-size:0.98em;margin-bottom:1em;'>BPR is the main ranking model used for final recommendations.</div>", unsafe_allow_html=True)

    # ===== FINAL RECOMMENDATIONS =====
    st.markdown("<h4 style='margin-top:0.5em;margin-bottom:0.5em;'>Final Recommendations</h4>", unsafe_allow_html=True)
    bpr_recs = get_bpr_recommendations(user_id, top_k, bpr_bundle)
    if bpr_recs:
        bpr_df = format_recommendations(bpr_recs, None, movies_df)
        st.dataframe(
            bpr_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.Column("Rank", width="small"),
                "Title": st.column_config.Column("Title", width="large"),
                "Score": st.column_config.Column("Score", width="small"),
            },
        )
    else:
        st.warning("No recommendations available for this user.")

# Remove any leftover code that references recommend_with_content or recommend_with_svd

    # ===== WATCHED HISTORY EXPANDER =====
    if show_history:
        with st.expander("Watched history", expanded=False):
            if not user_history.empty:
                st.dataframe(user_history[["Movie ID", "Title", "User Rating"]].head(20), use_container_width=True, hide_index=True)
            else:
                st.info("No movie watching history found for this user.")

    # ===== MODEL INFO EXPANDER =====
    if show_model_info:
        with st.expander("Model information", expanded=False):
            st.markdown("**BPR (Bayesian Personalized Ranking)** is the main production model powering these recommendations.")
            st.markdown("Other models (SVD, Content-Based, Baseline) are used for offline evaluation and benchmarking only.")


if __name__ == "__main__":
    main()
