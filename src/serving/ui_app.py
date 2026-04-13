"""Streamlit UI for movie recommendations using BPR and Content-Based models."""

from __future__ import annotations




import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd  # type: ignore[import-not-found]
import streamlit as st
from src.models.bpr import recommend_with_bundle as recommend_with_bpr
from src.models.content_based import recommend_with_bundle as recommend_with_content

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


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
def load_movie_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load movie metadata for display."""
    if metadata_path.exists():
        return pd.read_parquet(metadata_path)
    return pd.DataFrame({"movie_id": []})


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
        return {"movie_id": movie_id, "title": "Unknown", "genre": "N/A"}
    
    movie_row = movie.iloc[0]
    return {
        "movie_id": int(movie_row.get("movie_id", movie_id)),
        "title": str(movie_row.get("title", "Unknown")),
        "genre": str(movie_row.get("genre", "N/A")),
        "year": movie_row.get("year", "N/A"),
        "imdb_id": movie_row.get("imdb_id", "N/A"),
    }


def format_recommendations(
    recommendations: List[int],
    scores: List[float] | None,
    movies_df: pd.DataFrame,
    strategy: str,
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
            "Genre": movie_info["genre"],
            "Year": movie_info["year"],
            "Score": f"{score:.4f}" if score is not None else "N/A",
            "Strategy": strategy,
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


def compare_models(
    user_id: int,
    top_k: int,
    bpr_bundle: Dict,
    content_bundle: Dict,
    movies_df: pd.DataFrame,
) -> None:
    """Compare recommendations from both models side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 BPR Model")
        bpr_recs, bpr_scores, _ = get_bpr_recommendations(user_id, top_k, bpr_bundle)
        if bpr_recs:
            bpr_df = format_recommendations(bpr_recs, bpr_scores, movies_df, "BPR")
            st.dataframe(bpr_df, use_container_width=True)
        else:
            st.warning("No recommendations from BPR model")
    
    with col2:
        st.subheader("📚 Content-Based Model")
        content_recs, content_scores, _ = get_content_based_recommendations(user_id, top_k, content_bundle)
        if content_recs:
            content_df = format_recommendations(content_recs, content_scores, movies_df, "Content-Based")
            st.dataframe(content_df, use_container_width=True)
        else:
            st.warning("No recommendations from Content-Based model")


def show_ensemble_recommendations(
    user_id: int,
    top_k: int,
    bpr_bundle: Dict,
    content_bundle: Dict,
    movies_df: pd.DataFrame,
    ensemble_weight_bpr: float = 0.5,
) -> None:
    """Show ensemble recommendations combining both models."""
    st.subheader("🎯 Ensemble Recommendations")
    
    bpr_recs, bpr_scores, _ = get_bpr_recommendations(user_id, top_k * 2, bpr_bundle)
    content_recs, content_scores, _ = get_content_based_recommendations(user_id, top_k * 2, content_bundle)
    
    if not bpr_recs and not content_recs:
        st.warning("No recommendations available from either model")
        return
    
    # Create scoring dictionary
    ensemble_scores: Dict[int, float] = {}
    
    # Add BPR scores
    for i, movie_id in enumerate(bpr_recs):
        score = bpr_scores[i] if i < len(bpr_scores) else 0
        ensemble_scores[movie_id] = ensemble_scores.get(movie_id, 0) + (score * ensemble_weight_bpr)
    
    # Add Content-Based scores
    ensemble_weight_content = 1 - ensemble_weight_bpr
    for i, movie_id in enumerate(content_recs):
        score = content_scores[i] if i < len(content_scores) else 0
        ensemble_scores[movie_id] = ensemble_scores.get(movie_id, 0) + (score * ensemble_weight_content)
    
    # Sort by ensemble score
    sorted_movies = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
    ensemble_recommendations = [movie_id for movie_id, _ in sorted_movies[:top_k]]
    ensemble_scores_list = [score for _, score in sorted_movies[:top_k]]
    
    ensemble_df = format_recommendations(ensemble_recommendations, ensemble_scores_list, movies_df, "Ensemble")
    st.dataframe(ensemble_df, use_container_width=True)


def show_model_info(bpr_bundle: Dict, content_bundle: Dict) -> None:
    """Display model information and statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 BPR Model Info")
        if bpr_bundle:
            config = bpr_bundle.get("config", {})
            st.write(f"**Algorithm:** {bpr_bundle.get('algorithm', 'Unknown')}")
            st.write("**Model Config:**")
            for key, value in config.items():
                st.write(f"  - {key}: {value}")
    
    with col2:
        st.subheader("📊 Content-Based Model Info")
        if content_bundle:
            config = content_bundle.get("config", {})
            st.write(f"**Algorithm:** {content_bundle.get('algorithm', 'Unknown')}")
            st.write("**Model Config:**")
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
    
    st.title("🎬 Movie Recommender System")
    st.write("Compare recommendations from **BPR** and **Content-Based** models")
    
    # Load project paths
    project_root = get_project_root()
    bpr_model_path = project_root / "models" / "personalized" / "bpr_model.pkl"
    content_model_path = project_root / "models" / "personalized" / "content_based_model.pkl"
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        bpr_bundle = load_bpr_model(bpr_model_path)
        content_bundle = load_content_based_model(content_model_path)
        movies_df = load_movies_data()
    
    if not bpr_bundle and not content_bundle:
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
        ["Compare Models", "Ensemble", "BPR Only", "Content-Based Only"],
    )
    
    if recommendation_mode == "Ensemble":
        ensemble_weight_bpr = st.sidebar.slider(
            "BPR Model Weight (Ensemble):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
    
    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Model Info")
    show_model_info(bpr_bundle, content_bundle)
    
    # Main content
    st.markdown("---")
    
    if recommendation_mode == "Compare Models":
        compare_models(user_id, top_k, bpr_bundle, content_bundle, movies_df)
    
    elif recommendation_mode == "Ensemble":
        show_ensemble_recommendations(
            user_id,
            top_k,
            bpr_bundle,
            content_bundle,
            movies_df,
            ensemble_weight_bpr,
        )
    
    elif recommendation_mode == "BPR Only":
        st.subheader("🤖 BPR Model Recommendations")
        bpr_recs, bpr_scores, _ = get_bpr_recommendations(user_id, top_k, bpr_bundle)
        if bpr_recs:
            bpr_df = format_recommendations(bpr_recs, bpr_scores, movies_df, "BPR")
            st.dataframe(bpr_df, use_container_width=True)
        else:
            st.warning("No recommendations available")
    
    elif recommendation_mode == "Content-Based Only":
        st.subheader("📚 Content-Based Model Recommendations")
        content_recs, content_scores, _ = get_content_based_recommendations(user_id, top_k, content_bundle)
        if content_recs:
            content_df = format_recommendations(content_recs, content_scores, movies_df, "Content-Based")
            st.dataframe(content_df, use_container_width=True)
        else:
            st.warning("No recommendations available")


if __name__ == "__main__":
    main()
