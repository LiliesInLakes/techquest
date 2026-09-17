"""
src/evaluation/metrics.py

Pure mathematical implementations for evaluation metrics:
  - Ranking Quality: NDCG@K, HitRate@K
  - Quality Floor: Average Rating@K, Average Rating Count@K, Rating Coverage
  - Diversity: Intra-List Distance ILD@K
  - Personalization / History Coherence: Profile-Movie Content Similarity
  - Dual Coverage Tracking: History Input Coverage & Recommendation Feature Coverage
  - Offline Embedding Diagnostic: MiniLM 384d Cosine Similarity (Isolated Artifact Check)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D or 2D numpy arrays with zero-norm guards."""
    if vec1 is None or vec2 is None:
        return 0.0
    
    v1 = np.asarray(vec1).flatten()
    v2 = np.asarray(vec2).flatten()
    
    if v1.shape != v2.shape:
        return 0.0
        
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_hitrate(
    recommended_ids: List[int],
    expected_relevance_map: Dict[int, float],
    k: int
) -> float:
    """
    HitRate@K: Returns 1.0 if any of the top-K recommendations is present in
    expected_relevance_map with relevance > 0, else 0.0.
    """
    top_k_ids = recommended_ids[:k]
    for mid in top_k_ids:
        if expected_relevance_map.get(mid, 0) > 0:
            return 1.0
    return 0.0


def compute_ndcg(
    recommended_ids: List[int],
    expected_relevance_map: Dict[int, float],
    k: int
) -> Optional[float]:
    """
    NDCG@K: Graded relevance DCG@K / IDCG@K.
    IDCG is computed from ideal ground truth relevance scores sorted descending and truncated at K.
    Returns None if IDCG == 0 (no relevant items defined).
    """
    if not expected_relevance_map:
        return None
        
    # Ideal DCG@K from ground truth
    ideal_rels = sorted([r for r in expected_relevance_map.values() if r > 0], reverse=True)[:k]
    if not ideal_rels:
        return None
        
    idcg = sum((2.0 ** r - 1.0) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    if idcg == 0.0:
        return None
        
    # Actual DCG@K
    top_k_ids = recommended_ids[:k]
    dcg = 0.0
    for i, mid in enumerate(top_k_ids):
        rel = expected_relevance_map.get(mid, 0.0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
            
    return float(dcg / idcg)


def compute_quality_floor(
    recommended_movies: List[Dict[str, Any]],
    k: int
) -> Dict[str, Optional[float]]:
    """
    Computes Average Rating@K, Average Rating Count@K, and Rating Coverage@K.
    Movies with vote_count == 0 or missing vote_average are excluded from rating average.
    """
    top_k_movies = recommended_movies[:k]
    if not top_k_movies:
        return {
            "avg_rating": None,
            "avg_vote_count": None,
            "rating_coverage": 0.0
        }
        
    ratings = []
    vote_counts = []
    
    for m in top_k_movies:
        vote_avg = m.get("vote_average")
        vote_cnt = m.get("vote_count", 0)
        
        if vote_avg is not None and vote_cnt > 0 and float(vote_avg) > 0.0:
            ratings.append(float(vote_avg))
        if vote_cnt is not None:
            vote_counts.append(float(vote_cnt))
            
    num_total = len(top_k_movies)
    rating_cov = len(ratings) / num_total if num_total > 0 else 0.0
    avg_rating = float(np.mean(ratings)) if ratings else None
    avg_vote_count = float(np.mean(vote_counts)) if vote_counts else None
    
    return {
        "avg_rating": avg_rating,
        "avg_vote_count": avg_vote_count,
        "rating_coverage": rating_cov
    }


def compute_ild(
    recommended_ids: List[int],
    feature_matrix: Any,
    k: int
) -> Tuple[Optional[float], int]:
    """
    Intra-List Distance (ILD@K): Pairwise cosine distance among unique recommended movies
    present in FeatureMatrix.
    Returns (ild_score, num_valid_embeddings).
    """
    if feature_matrix is None or not hasattr(feature_matrix, "movie_ids"):
        return None, 0
        
    top_k_ids = recommended_ids[:k]
    unique_ids = list(dict.fromkeys(top_k_ids))
    
    embeddings = []
    for mid in unique_ids:
        if mid in feature_matrix.movie_ids:
            idx = feature_matrix.movie_ids.index(mid)
            emb = feature_matrix.semantic_embeddings[idx]
            embeddings.append(emb)
            
    n = len(embeddings)
    if n < 2:
        return (0.0 if n == 1 else None), n
        
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_cosine_similarity(embeddings[i], embeddings[j])
            distances.append(1.0 - sim)
            
    ild_val = float(np.mean(distances)) if distances else 0.0
    return ild_val, n


def compute_personalization(
    recommended_ids: List[int],
    user_profile: Any,
    preference_extractor: Any,
    feature_matrix: Any,
    k: int
) -> Tuple[Optional[float], int]:
    """
    Personalization / History Coherence Score@K:
    Average content similarity between user profile and top-K recommended movies.
    Returns (personalization_score, num_evaluated_movies).
    """
    if user_profile is None or preference_extractor is None or feature_matrix is None:
        return None, 0
        
    top_k_ids = recommended_ids[:k]
    if not top_k_ids:
        return None, 0
        
    sims = []
    for mid in top_k_ids:
        if mid in feature_matrix.movie_ids:
            sim = preference_extractor.compute_content_similarity(user_profile, mid, feature_matrix)
            sims.append(sim)
            
    if not sims:
        return None, 0
        
    return float(np.mean(sims)), len(sims)


def compute_dual_coverage(
    watch_history: List[Dict[str, Any]],
    recommended_ids: List[int],
    feature_matrix: Any,
    k: int
) -> Dict[str, float]:
    """
    Computes history_input_coverage and recommendation_feature_coverage_at_k.
    """
    # 1. History Input Coverage
    if not watch_history:
        history_cov = 1.0  # N/A or empty history = 100% of available
    else:
        hist_ids = [m.get("movie_id") or m.get("id") for m in watch_history if (m.get("movie_id") or m.get("id")) is not None]
        if not hist_ids:
            history_cov = 1.0
        else:
            in_fm = sum(1 for mid in hist_ids if feature_matrix and mid in feature_matrix.movie_ids)
            history_cov = float(in_fm / len(hist_ids))
            
    # 2. Recommendation Feature Coverage
    top_k_ids = recommended_ids[:k]
    if not top_k_ids:
        reco_cov = 0.0
    else:
        in_fm_reco = sum(1 for mid in top_k_ids if feature_matrix and mid in feature_matrix.movie_ids)
        reco_cov = float(in_fm_reco / len(top_k_ids))
        
    return {
        "history_input_coverage": history_cov,
        "recommendation_feature_coverage": reco_cov
    }


def compute_offline_embedding_diagnostic(
    query_text: str,
    recommended_ids: List[int],
    feature_matrix: Any,
    encoder_model_384: Any,
    k: int
) -> Tuple[Optional[float], int]:
    """
    Offline Semantic Embedding Diagnostic:
    Encodes query text using 384-dim SentenceTransformer ('all-MiniLM-L6-v2')
    and computes cosine similarity against stored movie embeddings in FeatureMatrix.
    
    THIS IS STRICTLY AN OFFLINE DIAGNOSTIC OF THE FEATURE MATRIX ARTIFACT, NOT RUNTIME RECOMMENDER BEHAVIOR.
    """
    if feature_matrix is None or encoder_model_384 is None or not query_text:
        return None, 0
        
    top_k_ids = recommended_ids[:k]
    if not top_k_ids:
        return None, 0
        
    # Encode query with 384d model
    query_emb_384 = encoder_model_384.encode([query_text])[0]
    
    sims = []
    for mid in top_k_ids:
        if mid in feature_matrix.movie_ids:
            idx = feature_matrix.movie_ids.index(mid)
            movie_emb_384 = feature_matrix.semantic_embeddings[idx]
            sim = compute_cosine_similarity(query_emb_384, movie_emb_384)
            sims.append(sim)
            
    if not sims:
        return None, 0
        
    return float(np.mean(sims)), len(sims)
