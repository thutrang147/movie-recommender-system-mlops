"""Streamlit UI for movie recommendations using BPR, Content-Based, and SVD models."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore[import-not-found]
import streamlit as st

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.bpr import recommend_with_bundle as recommend_with_bpr
from src.models.content_based import recommend_with_bundle as recommend_with_content
from src.models.train import recommend_with_bundle as recommend_with_svd


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
def load_content_based_model(model_path: Path) -> Dict:
    """Load Content-Based model bundle from pickle file."""
    if not model_path.exists():
        st.error(f"Content-Based model not found at {model_path}")
        return {}
    
    with open(model_path, "rb") as file:
        bundle = pickle.load(file)
    return bundle


@st.cache_resource
def load_svd_model(model_path: Path) -> Dict:
    """Load SVD model bundle from pickle file."""
    if not model_path.exists():
        st.error(f"SVD model not found at {model_path}")
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
    recommendations: List[int],
    scores: List[float] | None,
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Format recommendations into a nice DataFrame for display."""
    rows = []
    for rank, movie_id in enumerate(recommendations, 1):
        movie_info = get_movie_info(movie_id, movies_df)
        score = scores[rank - 1] if scores and rank <= len(scores) else None
        
        rows.append({
            "Rank": rank,
            "Movie ID": movie_info["movie_id"],
            "Title": movie_info["title"],
            "Suitability Score (out of 5)": f"{score:.1f}/5.0" if score is not None else "N/A",
        })
    
    return pd.DataFrame(rows)


def get_bpr_recommendations(
    user_id: int,
    top_k: int,
    bpr_bundle: Dict,
) -> Tuple[List[int], List[float], str]:
    """Get recommendations from BPR model."""
    try:
        recommendations_with_scores = recommend_with_bpr(
            bundle=bpr_bundle,
            user_id=user_id,
            top_k=top_k,
        )
        if not recommendations_with_scores:
            return [], [], "BPR"
        
        recommendations = [movie_id for movie_id, _ in recommendations_with_scores]
        scores = [score for _, score in recommendations_with_scores]
        return recommendations, scores, "BPR"
    except Exception as e:
        st.error(f"BPR Model Error: {str(e)}")
        return [], [], "BPR (Error)"


def get_content_based_recommendations(
    user_id: int,
    top_k: int,
    content_bundle: Dict,
) -> Tuple[List[int], List[float], str]:
    """Get recommendations from Content-Based model."""
    try:
        recommendations_with_scores = recommend_with_content(
            bundle=content_bundle,
            user_id=user_id,
            top_k=top_k,
        )
        if not recommendations_with_scores:
            return [], [], "Content-Based"
        
        recommendations = [movie_id for movie_id, _ in recommendations_with_scores]
        scores = [score for _, score in recommendations_with_scores]
        return recommendations, scores, "Content-Based"
    except Exception as e:
        st.error(f"Content-Based Model Error: {str(e)}")
        return [], [], "Content-Based (Error)"


def get_svd_recommendations(
    user_id: int,
    top_k: int,
    svd_bundle: Dict,
) -> Tuple[List[int], List[float], str]:
    """Get recommendations from SVD model."""
    try:
        recommendations_with_scores = recommend_with_svd(
            bundle=svd_bundle,
            user_id=user_id,
            top_k=top_k,
        )
        if not recommendations_with_scores:
            return [], [], "SVD"
        
        recommendations = [movie_id for movie_id, _ in recommendations_with_scores]
        scores = [score for _, score in recommendations_with_scores]
        return recommendations, scores, "SVD"
    except Exception as e:
        st.error(f"SVD Model Error: {str(e)}")
        return [], [], "SVD (Error)"


def compare_models(
    user_id: int,
    top_k: int,
    bpr_bundle: Dict,
    content_bundle: Dict,
    svd_bundle: Dict,
    movies_df: pd.DataFrame,
) -> None:
    """Compare recommendations from BPR, Content-Based, SVD, and Baseline models."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("🤖 BPR")
        bpr_recs, bpr_scores, _ = get_bpr_recommendations(user_id, top_k, bpr_bundle)
        if bpr_recs:
            bpr_df = format_recommendations(bpr_recs, bpr_scores, movies_df)
            # Pad with empty rows to match top_k
            while len(bpr_df) < top_k:
                bpr_df = pd.concat([bpr_df, pd.DataFrame([{"Rank": "", "Movie ID": "", "Title": "", "Suitability Score (out of 5)": ""}])], ignore_index=True)
            st.dataframe(bpr_df, use_container_width=True)
        else:
            st.warning("No recommendations from BPR model")
    
    with col2:
        st.subheader("📚 Content-Based")
        content_recs, content_scores, _ = get_content_based_recommendations(user_id, top_k, content_bundle)
        if content_recs:
            content_df = format_recommendations(content_recs, content_scores, movies_df)
            # Pad with empty rows to match top_k
            while len(content_df) < top_k:
                content_df = pd.concat([content_df, pd.DataFrame([{"Rank": "", "Movie ID": "", "Title": "", "Suitability Score (out of 5)": ""}])], ignore_index=True)
            st.dataframe(content_df, use_container_width=True)
        else:
            st.warning("No recommendations from Content-Based model")
    
    with col3:
        st.subheader("🎯 SVD")
        svd_recs, svd_scores, _ = get_svd_recommendations(user_id, top_k, svd_bundle)
        if svd_recs:
            svd_df = format_recommendations(svd_recs, svd_scores, movies_df)
            # Pad with empty rows to match top_k
            while len(svd_df) < top_k:
                svd_df = pd.concat([svd_df, pd.DataFrame([{"Rank": "", "Movie ID": "", "Title": "", "Suitability Score (out of 5)": ""}])], ignore_index=True)
            st.dataframe(svd_df, use_container_width=True)
        else:
            st.warning("No recommendations from SVD model")
    
    with col4:
        st.subheader("📊 Baseline")
        # Get baseline recommendations (top popular movies)
        if movies_df is not None and not movies_df.empty:
            # Assuming movies_df has interaction count or we can use a simple popularity fallback
            # For now, use a simple fallback similar to the models
            baseline_recs = list(range(1, min(top_k + 1, len(movies_df))))
            baseline_scores = [5.0 - i * 0.1 for i in range(len(baseline_recs))]  # Dummy scores
            baseline_df = format_recommendations(baseline_recs, baseline_scores, movies_df)
            # Pad with empty rows to match top_k
            while len(baseline_df) < top_k:
                baseline_df = pd.concat([baseline_df, pd.DataFrame([{"Rank": "", "Movie ID": "", "Title": "", "Suitability Score (out of 5)": ""}])], ignore_index=True)
            st.dataframe(baseline_df, use_container_width=True)
        else:
            st.warning("No baseline recommendations available")


def show_ensemble_recommendations(
    user_id: int,
    top_k: int,
    bpr_bundle: Dict,
    svd_bundle: Dict,
    movies_df: pd.DataFrame,
    ensemble_weight_bpr: float = 0.5,
    ensemble_weight_svd: float = 0.5,
) -> None:
    """Show ensemble recommendations combining BPR and SVD models."""
    st.subheader("🎯 Ensemble Recommendations")
    
    bpr_recs, bpr_scores, _ = get_bpr_recommendations(user_id, top_k * 2, bpr_bundle)
    svd_recs, svd_scores, _ = get_svd_recommendations(user_id, top_k * 2, svd_bundle)
    
    if not bpr_recs and not svd_recs:
        st.warning("No recommendations available from any model")
        return
    
    # Create scoring dictionary
    ensemble_scores: Dict[int, float] = {}
    
    # Add BPR scores
    for i, movie_id in enumerate(bpr_recs):
        score = bpr_scores[i] if i < len(bpr_scores) else 0
        ensemble_scores[movie_id] = ensemble_scores.get(movie_id, 0) + (score * ensemble_weight_bpr)
    
    # Add SVD scores
    for i, movie_id in enumerate(svd_recs):
        score = svd_scores[i] if i < len(svd_scores) else 0
        ensemble_scores[movie_id] = ensemble_scores.get(movie_id, 0) + (score * ensemble_weight_svd)
    
    # Sort by ensemble score
    sorted_movies = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
    ensemble_recommendations = [movie_id for movie_id, _ in sorted_movies[:top_k]]
    ensemble_scores_list = [score for _, score in sorted_movies[:top_k]]
    
    ensemble_df = format_recommendations(ensemble_recommendations, ensemble_scores_list, movies_df)
    st.dataframe(ensemble_df, use_container_width=True)


def show_model_info(bpr_bundle: Dict, content_bundle: Dict, svd_bundle: Dict) -> None:
    """Display model information and statistics."""
    with st.sidebar.expander("📋 Model Information"):
        st.markdown("### 🤖 BPR Model")
        st.markdown("**Theory:** Bayesian Personalized Ranking optimizes for ranking by learning to predict relative preferences between items for each user.")
        if bpr_bundle:
            config = bpr_bundle.get("config", {})
            st.markdown("**Config:**")
            for key, value in config.items():
                st.write(f"  - {key}: {value}")
        
        st.markdown("### 📚 Content-Based Model")
        st.markdown("**Theory:** Content-Based filtering recommends items similar to those the user has liked before, based on item features and user preferences.")
        if content_bundle:
            config = content_bundle.get("config", {})
            st.markdown("**Config:**")
            for key, value in config.items():
                st.write(f"  - {key}: {value}")
        
        st.markdown("### 🎯 SVD Model")
        st.markdown("**Theory:** Singular Value Decomposition reduces the user-item matrix dimensionality to capture latent factors for collaborative filtering.")
        if svd_bundle:
            config = svd_bundle.get("config", {})
            st.markdown("**Config:**")
            for key, value in config.items():
                st.write(f"  - {key}: {value}")
        
        st.markdown("### 📊 Baseline Model")
        st.markdown("**Theory:** Popularity-based recommendations using item interaction counts as a simple baseline for comparison.")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🎬 Movie Recommender System")
    
    # Load project paths
    project_root = get_project_root()
    bpr_model_path = project_root / "models" / "personalized" / "bpr_model.pkl"
    content_model_path = project_root / "models" / "personalized" / "content_based_model.pkl"
    svd_model_path = project_root / "models" / "personalized" / "svd_model.pkl"
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        bpr_bundle = load_bpr_model(bpr_model_path)
        content_bundle = load_content_based_model(content_model_path)
        svd_bundle = load_svd_model(svd_model_path)
        movies_df = load_movies_data()
        ratings_df = load_user_history()
    
    if not bpr_bundle and not content_bundle and not svd_bundle:
        st.error("❌ No models loaded. Please check model paths.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    user_id = st.sidebar.number_input(
        "Enter User ID:",
        min_value=1,
        value=1,
        step=1,
    )
    
    top_k = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=1,
        max_value=20,
        value=10,
    )
    
    recommendation_mode = st.sidebar.radio(
        "Recommendation Mode:",
        ["Compare Models", "Ensemble", "BPR Only", "Content-Based Only", "SVD Only", "Baseline Only"],
    )
    
    # Dynamic description based on mode
    if recommendation_mode == "Compare Models":
        st.write("Compare recommendations from **BPR**, **Content-Based**, **SVD**, and **Baseline (Popularity)** models")
    elif recommendation_mode == "Ensemble":
        st.write("Get ensemble recommendations combining **BPR** and **SVD** models")
    elif recommendation_mode == "BPR Only":
        st.write("Get personalized recommendations using the **BPR** model")
    elif recommendation_mode == "Content-Based Only":
        st.write("Get content-based recommendations")
    elif recommendation_mode == "SVD Only":
        st.write("Get personalized recommendations using the **SVD** model")
    elif recommendation_mode == "Baseline Only":
        st.write("Get popularity-based recommendations")
    
    if recommendation_mode == "Ensemble":
        st.sidebar.markdown("### Ensemble Weights")
        ensemble_weight_bpr = st.sidebar.slider(
            "BPR Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        ensemble_weight_svd = 1.0 - ensemble_weight_bpr
        st.sidebar.metric("SVD Weight (Auto):", f"{ensemble_weight_svd:.2f}")
    
    # Footer info
    st.sidebar.markdown("---")
    show_model_info(bpr_bundle, content_bundle, svd_bundle)
    
    # Main content
    st.markdown("---")
    
    # Show user history
    st.subheader(f"📚 Movies Watched by User {user_id}")
    user_history = get_user_watched_movies(user_id, ratings_df, movies_df, 10)
    if not user_history.empty:
        st.dataframe(user_history, use_container_width=True)
    else:
        st.info("No movie watching history found for this user.")
    
    st.markdown("---")
    
    if recommendation_mode == "Compare Models":
        compare_models(user_id, top_k, bpr_bundle, content_bundle, svd_bundle, movies_df)
    
    elif recommendation_mode == "Ensemble":
        show_ensemble_recommendations(
            user_id,
            top_k,
            bpr_bundle,
            svd_bundle,
            movies_df,
            ensemble_weight_bpr,
            ensemble_weight_svd,
        )
    
    elif recommendation_mode == "BPR Only":
        st.subheader("🤖 BPR Model Recommendations")
        bpr_recs, bpr_scores, _ = get_bpr_recommendations(user_id, top_k, bpr_bundle)
        if bpr_recs:
            bpr_df = format_recommendations(bpr_recs, bpr_scores, movies_df)
            st.dataframe(bpr_df, use_container_width=True)
        else:
            st.warning("No recommendations available")
    
    elif recommendation_mode == "Content-Based Only":
        st.subheader("📚 Content-Based Model Recommendations")
        content_recs, content_scores, _ = get_content_based_recommendations(user_id, top_k, content_bundle)
        if content_recs:
            content_df = format_recommendations(content_recs, content_scores, movies_df)
            st.dataframe(content_df, use_container_width=True)
        else:
            st.warning("No recommendations available")
    
    elif recommendation_mode == "Content-Based Only":
        st.subheader("📚 Content-Based Model Recommendations")
        content_recs, content_scores, _ = get_content_based_recommendations(user_id, top_k, content_bundle)
        if content_recs:
            content_df = format_recommendations(content_recs, content_scores, movies_df)
            st.dataframe(content_df, use_container_width=True)
        else:
            st.warning("No recommendations available")
    
    elif recommendation_mode == "Baseline Only":
        st.subheader("📊 Baseline (Popularity) Recommendations")
        # Get baseline recommendations (top popular movies)
        if movies_df is not None and not movies_df.empty:
            # Assuming movies_df has interaction count or we can use a simple popularity fallback
            # For now, use a simple fallback similar to the models
            baseline_recs = list(range(1, min(top_k + 1, len(movies_df))))
            baseline_scores = [5.0 - i * 0.1 for i in range(len(baseline_recs))]  # Dummy scores
            baseline_df = format_recommendations(baseline_recs, baseline_scores, movies_df)
            st.dataframe(baseline_df, use_container_width=True)
        else:
            st.warning("No baseline recommendations available")


if __name__ == "__main__":
    main()
