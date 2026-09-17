import numpy as np
import math
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import yaml


# ==========================================
# Explicit Ranking Constants
# ==========================================
RATING_WEIGHT = 0.30
TFIDF_WEIGHT = 0.25
SEMANTIC_WEIGHT = 0.35
POPULARITY_WEIGHT = 0.10

PRIMARY_COHORT_RADIUS = 3
SECONDARY_COHORT_RADIUS = 5

COHORT_POPULARITY_WEIGHT = 0.70
GLOBAL_POPULARITY_WEIGHT = 0.30

MIN_COHORT_SIZE = 20


@dataclass
class RankingConfig:
    imdb_weight: float = RATING_WEIGHT
    tfidf_weight: float = TFIDF_WEIGHT
    semantic_weight: float = SEMANTIC_WEIGHT
    popularity_weight: float = POPULARITY_WEIGHT
    watch_history_weight: float = 0.5
    content_weight: float = 0.4
    keyword_overlap_weight: float = 0.3
    min_vote_count: int = 100
    popularity_boost: float = 0.1  # retained for backward compatibility


@dataclass
class RankedMovie:
    movie_id: int
    title: str
    score: float
    imdb_score: float
    tfidf_score: float
    watch_history_score: float
    semantic_score: float
    content_score: float
    keyword_overlap_score: float
    popularity_score: float
    overview: str
    genres: List[str]
    vote_average: float
    vote_count: int
    reasoning_factors: Dict[str, float]


class MovieRanker:
    def __init__(self, config: RankingConfig = None, config_path: str = "config.yaml"):
        self.config = config or self._load_config(config_path)
        self.preference_extractor = None
        self.feature_matrix = None
        self.semantic_search = None
        self._cohort_index_built = False
        self._global_max_rating_count = 102929
        self._global_log_max = math.log1p(102929)
        self._cohort_cache_r3: Dict[int, np.ndarray] = {}
        self._cohort_cache_r5: Dict[int, np.ndarray] = {}

    def _load_config(self, config_path: str) -> RankingConfig:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            rec_cfg = cfg.get("recommendation", {}) if cfg else {}
            return RankingConfig(
                imdb_weight=float(rec_cfg.get("imdb_weight", RATING_WEIGHT)),
                tfidf_weight=float(rec_cfg.get("tfidf_weight", TFIDF_WEIGHT)),
                semantic_weight=float(rec_cfg.get("semantic_weight", SEMANTIC_WEIGHT)),
                popularity_weight=float(rec_cfg.get("popularity_weight", POPULARITY_WEIGHT)),
                watch_history_weight=float(rec_cfg.get("watch_history_weight", 0.5)),
                content_weight=0.4,
                keyword_overlap_weight=0.3
            )
        except (FileNotFoundError, Exception):
            return RankingConfig()

    def set_dependencies(self, preference_extractor, feature_matrix, semantic_search):
        self.preference_extractor = preference_extractor
        self.feature_matrix = feature_matrix
        self.semantic_search = semantic_search
        self._cohort_index_built = False
        self._build_cohort_index()

    def rank_movies(
        self,
        candidate_movies: List[Dict[str, Any]],
        user_profile,
        query_embedding: np.ndarray = None,
        query_tfidf: csr_matrix = None
    ) -> List[RankedMovie]:
        if not candidate_movies:
            return []
        
        ranked = []
        
        for movie in candidate_movies:
            movie_id = movie.get('id') or movie.get('movie_id')
            if not movie_id or movie_id not in self.feature_matrix.movie_metadata:
                continue
            
            scores = self._compute_all_scores(movie, user_profile, query_embedding, query_tfidf)
            
            final_score = self._compute_final_score(scores)
            
            ranked.append(RankedMovie(
                movie_id=movie_id,
                title=movie.get('title', ''),
                score=final_score,
                imdb_score=scores['imdb'],
                tfidf_score=scores['tfidf'],
                watch_history_score=scores['watch_history'],
                semantic_score=scores['semantic'],
                content_score=scores['content'],
                keyword_overlap_score=scores['keyword_overlap'],
                popularity_score=scores['popularity'],
                overview=movie.get('overview', ''),
                genres=movie.get('genres', []),
                vote_average=movie.get('vote_average', 0),
                vote_count=movie.get('vote_count', 0),
                reasoning_factors=scores
            ))
        
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def rank_movies_stage1(
        self,
        candidate_movies: List[Dict[str, Any]],
        query_embedding: np.ndarray = None,
        query_tfidf: csr_matrix = None
    ) -> List[RankedMovie]:
        """Stage 1: Rank by query similarity + IMDb + popularity only (no watch history)"""
        if not candidate_movies:
            return []
        
        ranked = []
        
        for movie in candidate_movies:
            movie_id = movie.get('id') or movie.get('movie_id')
            if not movie_id:
                continue
            
            scores = {}
            scores['imdb'] = self._compute_imdb_score(movie)
            scores['tfidf'] = self._compute_tfidf_score(movie_id, query_tfidf, movie)
            scores['semantic'] = self._compute_semantic_score(movie_id, query_embedding, movie)
            scores['popularity'] = self._compute_popularity_score(movie)
            scores['watch_history'] = 0.0
            scores['content'] = 0.0
            scores['keyword_overlap'] = 0.0
            
            final_score = self._compute_stage1_score(scores)
            
            ranked.append(RankedMovie(
                movie_id=movie_id,
                title=movie.get('title', ''),
                score=final_score,
                imdb_score=scores['imdb'],
                tfidf_score=scores['tfidf'],
                watch_history_score=scores['watch_history'],
                semantic_score=scores['semantic'],
                content_score=scores['content'],
                keyword_overlap_score=scores['keyword_overlap'],
                popularity_score=scores['popularity'],
                overview=movie.get('overview', ''),
                genres=movie.get('genres', []),
                vote_average=movie.get('vote_average', 0),
                vote_count=movie.get('vote_count', 0),
                reasoning_factors=scores
            ))
        
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def rank_movies_stage2(
        self,
        stage1_ranked: List[RankedMovie],
        user_profile,
        extracted_features,
        feature_matrix,
        history_boost_factor: float = 0.3,
        candidate_lookup: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> List[RankedMovie]:
        """Stage 2: Apply history boost within matching genre clusters"""
        if not stage1_ranked or user_profile is None or self.preference_extractor is None:
            return stage1_ranked
        
        extracted_genres = set(g.lower() for g in extracted_features.genres) if extracted_features else set()
        
        boosted = []
        for rm in stage1_ranked:
            movie_genres = set(g.lower() for g in rm.genres)
            cand_meta = candidate_lookup.get(rm.movie_id, {}) if candidate_lookup else {}
            
            genre_match = bool(extracted_genres & movie_genres) if extracted_genres else True
            penalty = self.compute_negative_penalty(rm.movie_id, user_profile, feature_matrix, movie_dict=cand_meta)
            
            if genre_match:
                history_score = self.preference_extractor.compute_content_similarity(
                    user_profile, rm.movie_id, feature_matrix
                )
                keyword_overlap = self.preference_extractor.compute_keyword_overlap_score(
                    user_profile, rm.movie_id, feature_matrix
                )
                
                boost = history_boost_factor * history_score
                boost += history_boost_factor * 0.5 * keyword_overlap
                
                new_score = rm.score * (1.0 + boost) * penalty
                
                new_factors = rm.reasoning_factors.copy()
                new_factors['watch_history'] = history_score
                new_factors['keyword_overlap'] = keyword_overlap
                new_factors['history_boost'] = boost
                new_factors['negative_penalty'] = penalty
                
                boosted.append(RankedMovie(
                    movie_id=rm.movie_id,
                    title=rm.title,
                    score=new_score,
                    imdb_score=rm.imdb_score,
                    tfidf_score=rm.tfidf_score,
                    watch_history_score=history_score,
                    semantic_score=rm.semantic_score,
                    content_score=history_score,
                    keyword_overlap_score=keyword_overlap,
                    popularity_score=rm.popularity_score,
                    overview=rm.overview,
                    genres=rm.genres,
                    vote_average=rm.vote_average,
                    vote_count=rm.vote_count,
                    reasoning_factors=new_factors
                ))
            else:
                new_factors = rm.reasoning_factors.copy()
                new_factors['watch_history'] = 0.0
                new_factors['keyword_overlap'] = 0.0
                non_match_weight = 0.2 if extracted_genres else 1.0
                boosted.append(RankedMovie(
                    movie_id=rm.movie_id,
                    title=rm.title,
                    score=rm.score * non_match_weight * penalty,
                    imdb_score=rm.imdb_score,
                    tfidf_score=rm.tfidf_score,
                    watch_history_score=0.0,
                    semantic_score=rm.semantic_score,
                    content_score=0.0,
                    keyword_overlap_score=0.0,
                    popularity_score=rm.popularity_score,
                    overview=rm.overview,
                    genres=rm.genres,
                    vote_average=rm.vote_average,
                    vote_count=rm.vote_count,
                    reasoning_factors=new_factors
                ))
        
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def compute_negative_penalty(
        self,
        movie_id: int,
        user_profile,
        feature_matrix=None,
        movie_dict: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Computes blended negative penalty:
        penalty(movie) = 1.0 - min(0.9, 0.75 * max_{f in movie}(neg(f)) + 0.25 * (1/|F|) * sum_{f in movie} neg(f))
        """
        if user_profile is None or not hasattr(user_profile, "negative_preferences"):
            return 1.0
        neg_prefs = getattr(user_profile, "negative_preferences", {})
        if not neg_prefs:
            return 1.0

        fm = feature_matrix or self.feature_matrix
        meta = None
        if fm is not None and hasattr(fm, "movie_metadata") and movie_id in fm.movie_metadata:
            meta = fm.movie_metadata[movie_id]
        elif movie_dict:
            meta = movie_dict

        if not meta:
            return 1.0

        features = []
        for g in meta.get("genres", []):
            if isinstance(g, str):
                features.append(g.strip().lower())
            elif isinstance(g, dict) and "name" in g:
                features.append(g["name"].strip().lower())
        for kw in meta.get("keywords", []):
            if isinstance(kw, str):
                features.append(kw.strip().lower())
            elif isinstance(kw, dict) and "name" in kw:
                features.append(kw["name"].strip().lower())
        for d in meta.get("director", []):
            if isinstance(d, str):
                features.append(d.strip().lower())
        for c in meta.get("cast", [])[:5]:
            if isinstance(c, str):
                features.append(c.strip().lower())

        # Tags (MovieLens)
        ml_tags = meta.get("movielens_tags", "")
        if isinstance(ml_tags, str) and ml_tags.strip():
            features.extend([t.strip().lower() for t in ml_tags.split(",") if t.strip()][:15])
        elif isinstance(ml_tags, list):
            features.extend([str(t).strip().lower() for t in ml_tags[:15] if str(t).strip()])
        for tag in meta.get("tags", []):
            if isinstance(tag, str):
                features.append(tag.strip().lower())

        # Themes & Tones
        for th in meta.get("themes", meta.get("theme", [])):
            if isinstance(th, str):
                features.append(th.strip().lower())
        for tn in meta.get("tones", meta.get("tone", meta.get("mood", []))):
            if isinstance(tn, str):
                features.append(tn.strip().lower())

        if not features:
            return 1.0

        neg_scores = [neg_prefs.get(f, 0.0) for f in features]
        max_neg = max(neg_scores) if neg_scores else 0.0
        avg_neg = (sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0

        if max_neg == 0.0 and avg_neg == 0.0:
            return 1.0

        raw_penalty = 0.75 * max_neg + 0.25 * avg_neg
        penalty_factor = 1.0 - min(0.9, raw_penalty)
        return max(0.1, min(1.0, penalty_factor))

    def _compute_all_scores(
        self,
        movie: Dict[str, Any],
        user_profile,
        query_embedding: np.ndarray = None,
        query_tfidf: csr_matrix = None
    ) -> Dict[str, float]:
        movie_id = movie.get('id') or movie.get('movie_id')
        meta = self.feature_matrix.movie_metadata.get(movie_id, movie) if self.feature_matrix and hasattr(self.feature_matrix, "movie_metadata") else movie
        
        scores = {}
        
        scores['imdb'] = self._compute_imdb_score(movie)
        scores['tfidf'] = self._compute_tfidf_score(movie_id, query_tfidf, movie)
        scores['watch_history'] = self._compute_watch_history_score(movie_id, user_profile)
        scores['semantic'] = self._compute_semantic_score(movie_id, query_embedding, movie)
        scores['content'] = self._compute_content_score(movie_id, user_profile)
        scores['keyword_overlap'] = self._compute_keyword_overlap_score(movie_id, user_profile)
        scores['popularity'] = self._compute_popularity_score(movie)
        scores['negative_penalty'] = self.compute_negative_penalty(movie_id, user_profile, self.feature_matrix, movie_dict=movie)
        
        return scores

    def _build_cohort_index(self):
        """Precomputes release-year rating count distributions for fast cohort popularity lookup."""
        if self._cohort_index_built:
            return

        metadata = getattr(self.feature_matrix, "movie_metadata", {}) if self.feature_matrix else {}
        year_to_logs: Dict[int, List[float]] = {}
        all_counts: List[int] = []

        if metadata:
            for mid, meta in metadata.items():
                cnt = self._extract_rating_count(meta)
                all_counts.append(cnt)
                log_c = math.log1p(cnt)
                yr = self._extract_release_year(meta)
                if yr is not None:
                    year_to_logs.setdefault(yr, []).append(log_c)

        self._global_max_rating_count = max(all_counts) if all_counts else 102929
        self._global_log_max = math.log1p(self._global_max_rating_count)

        self._cohort_cache_r3 = {}
        self._cohort_cache_r5 = {}

        all_years = sorted(year_to_logs.keys())
        for yr in all_years:
            # Primary radius 3 (Y +/- 3)
            r3_logs = []
            for delta in range(-PRIMARY_COHORT_RADIUS, PRIMARY_COHORT_RADIUS + 1):
                r3_logs.extend(year_to_logs.get(yr + delta, []))
            if r3_logs:
                self._cohort_cache_r3[yr] = np.sort(np.array(r3_logs, dtype=np.float32))

            # Secondary radius 5 (Y +/- 5)
            r5_logs = []
            for delta in range(-SECONDARY_COHORT_RADIUS, SECONDARY_COHORT_RADIUS + 1):
                r5_logs.extend(year_to_logs.get(yr + delta, []))
            if r5_logs:
                self._cohort_cache_r5[yr] = np.sort(np.array(r5_logs, dtype=np.float32))

        self._cohort_index_built = True

    def _extract_release_year(self, movie: Dict[str, Any]) -> Optional[int]:
        if not movie or not isinstance(movie, dict):
            return None
        
        # 1. release_date (e.g. '1999-03-31')
        rd = movie.get('release_date')
        if rd and isinstance(rd, str) and len(rd.strip()) >= 4:
            try:
                yr = int(rd.strip()[:4])
                if 1880 <= yr <= 2100:
                    return yr
            except (ValueError, TypeError):
                pass
        
        # 2. release_year or year
        ry = movie.get('release_year') or movie.get('year')
        if ry is not None:
            try:
                yr = int(ry)
                if 1880 <= yr <= 2100:
                    return yr
            except (ValueError, TypeError):
                pass
        
        # 3. Title embedded year, e.g. "The Matrix (1999)"
        title = movie.get('title')
        if title and isinstance(title, str):
            match = re.search(r'\((\d{4})\)', title)
            if match:
                try:
                    yr = int(match.group(1))
                    if 1880 <= yr <= 2100:
                        return yr
                except (ValueError, TypeError):
                    pass
        
        # 4. Check feature_matrix metadata if movie_id present
        mid = movie.get('id') or movie.get('movie_id')
        if mid and self.feature_matrix and hasattr(self.feature_matrix, "movie_metadata"):
            meta = self.feature_matrix.movie_metadata.get(mid, {})
            if meta:
                meta_rd = meta.get('release_date')
                if meta_rd and isinstance(meta_rd, str) and len(meta_rd.strip()) >= 4:
                    try:
                        yr = int(meta_rd.strip()[:4])
                        if 1880 <= yr <= 2100:
                            return yr
                    except (ValueError, TypeError):
                        pass
                meta_title = meta.get('title')
                if meta_title and isinstance(meta_title, str):
                    match = re.search(r'\((\d{4})\)', meta_title)
                    if match:
                        try:
                            yr = int(match.group(1))
                            if 1880 <= yr <= 2100:
                                return yr
                        except (ValueError, TypeError):
                            pass

        return None

    def _extract_rating_count(self, movie: Dict[str, Any]) -> int:
        if not movie or not isinstance(movie, dict):
            return 0
        
        rc = movie.get('rating_count')
        if rc is not None:
            try:
                val = int(rc)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
        
        vc = movie.get('vote_count')
        if vc is not None:
            try:
                val = int(vc)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
        
        mid = movie.get('id') or movie.get('movie_id')
        if mid and self.feature_matrix and hasattr(self.feature_matrix, "movie_metadata"):
            meta = self.feature_matrix.movie_metadata.get(mid, {})
            if meta:
                m_rc = meta.get('rating_count') or meta.get('vote_count')
                if m_rc is not None:
                    try:
                        val = int(m_rc)
                        if val > 0:
                            return val
                    except (ValueError, TypeError):
                        pass

        return 0

    def _compute_stage1_score(self, scores: Dict[str, float]) -> float:
        w = self.config
        final = (
            scores['imdb'] * w.imdb_weight +
            scores['tfidf'] * w.tfidf_weight +
            scores['semantic'] * w.semantic_weight +
            scores['popularity'] * w.popularity_weight
        )
        return float(final)

    def _compute_imdb_score(self, movie: Dict[str, Any]) -> float:
        vote_avg = movie.get('vote_average', 0)
        vote_count = movie.get('vote_count', 0)
        
        if vote_count < self.config.min_vote_count:
            return vote_avg / 10.0 * 0.5
        
        return vote_avg / 10.0

    def _compute_tfidf_score(
        self,
        movie_id: int,
        query_tfidf: csr_matrix,
        movie: Optional[Dict[str, Any]] = None
    ) -> float:
        if query_tfidf is None:
            return 0.0
        
        try:
            movie_idx = None
            if self.feature_matrix is not None:
                if hasattr(self.feature_matrix, "tmdb_id_to_idx") and isinstance(self.feature_matrix.tmdb_id_to_idx, dict):
                    movie_idx = self.feature_matrix.tmdb_id_to_idx.get(movie_id)
                elif hasattr(self.feature_matrix, "movie_ids") and isinstance(self.feature_matrix.movie_ids, list):
                    movie_idx = self.feature_matrix.movie_ids.index(movie_id) if movie_id in self.feature_matrix.movie_ids else None
                
            if movie_idx is None or not isinstance(movie_idx, (int, np.integer)):
                if movie and 'tfidf_vector' in movie and movie['tfidf_vector'] is not None:
                    movie_tfidf = movie['tfidf_vector']
                    if hasattr(movie_tfidf, 'toarray'):
                        movie_tfidf = movie_tfidf.toarray()
                    sim = cosine_similarity(query_tfidf, movie_tfidf)[0][0]
                    return float(sim)
                return 0.0

            movie_tfidf = self.feature_matrix.tfidf_matrix[movie_idx]
            
            if hasattr(movie_tfidf, 'toarray'):
                movie_tfidf = movie_tfidf.toarray()
            
            sim = cosine_similarity(query_tfidf, movie_tfidf)[0][0]
            return float(sim)
        except (ValueError, IndexError):
            return 0.0

    def _compute_watch_history_score(self, movie_id: int, user_profile) -> float:
        if user_profile is None or self.preference_extractor is None:
            return 0.0
        
        return self.preference_extractor.compute_content_similarity(user_profile, movie_id, self.feature_matrix)

    def _compute_semantic_score(
        self,
        movie_id: int,
        query_embedding: np.ndarray,
        movie: Optional[Dict[str, Any]] = None
    ) -> float:
        if query_embedding is None:
            return 0.0
        
        try:
            movie_idx = None
            if self.feature_matrix is not None:
                if hasattr(self.feature_matrix, "tmdb_id_to_idx") and isinstance(self.feature_matrix.tmdb_id_to_idx, dict):
                    movie_idx = self.feature_matrix.tmdb_id_to_idx.get(movie_id)
                elif hasattr(self.feature_matrix, "movie_ids") and isinstance(self.feature_matrix.movie_ids, list):
                    movie_idx = self.feature_matrix.movie_ids.index(movie_id) if movie_id in self.feature_matrix.movie_ids else None
                
            if movie_idx is None or not isinstance(movie_idx, (int, np.integer)):
                if movie and 'semantic_embedding' in movie and movie['semantic_embedding'] is not None:
                    m_vec = np.asarray(movie['semantic_embedding']).flatten()
                else:
                    return 0.0
            else:
                m_vec = np.asarray(self.feature_matrix.semantic_embeddings[movie_idx]).flatten()

            q_vec = np.asarray(query_embedding).flatten()
            norm_q = np.linalg.norm(q_vec)
            norm_m = np.linalg.norm(m_vec)
            if norm_q == 0 or norm_m == 0:
                return 0.0
            sim = np.dot(q_vec, m_vec) / (norm_q * norm_m)
            return float(max(0.0, float(sim)))
        except (ValueError, IndexError, TypeError):
            return 0.0

    def _compute_content_score(self, movie_id: int, user_profile) -> float:
        if user_profile is None or self.preference_extractor is None:
            return 0.0
        
        return self.preference_extractor.compute_content_similarity(user_profile, movie_id, self.feature_matrix)

    def _compute_keyword_overlap_score(self, movie_id: int, user_profile) -> float:
        if user_profile is None or self.preference_extractor is None:
            return 0.0
        
        return self.preference_extractor.compute_keyword_overlap_score(user_profile, movie_id, self.feature_matrix)

    def _compute_popularity_score(self, movie: Dict[str, Any]) -> float:
        """
        Computes Age-Relative Popularity:
        age_adjusted_popularity = 0.70 * cohort_percentile + 0.30 * global_popularity
        - Uses log1p(rating_count)
        - Cohort radius: primary +/- 3 years, secondary +/- 5 years
        - Safely falls back to global popularity if year missing or cohort sparse (< 20 items)
        - Safely handles zero / NULL counts
        """
        self._build_cohort_index()

        rating_count = self._extract_rating_count(movie)
        if rating_count <= 0:
            return 0.0

        log_count = math.log1p(rating_count)
        global_pop = (log_count / self._global_log_max) if self._global_log_max > 0 else 0.0
        global_pop = max(0.0, min(1.0, global_pop))

        year = self._extract_release_year(movie)

        # Missing release year: safely fall back to global popularity
        if year is None:
            return float(global_pop)

        # Determine cohort
        cohort = None
        if year in self._cohort_cache_r3 and len(self._cohort_cache_r3[year]) >= MIN_COHORT_SIZE:
            cohort = self._cohort_cache_r3[year]
        elif year in self._cohort_cache_r5 and len(self._cohort_cache_r5[year]) >= MIN_COHORT_SIZE:
            cohort = self._cohort_cache_r5[year]

        if cohort is None or len(cohort) == 0:
            return float(global_pop)

        # Cohort percentile rank
        rank = np.searchsorted(cohort, log_count, side='right')
        cohort_percentile = float(rank / len(cohort))
        cohort_percentile = max(0.0, min(1.0, cohort_percentile))

        age_adjusted = (
            COHORT_POPULARITY_WEIGHT * cohort_percentile +
            GLOBAL_POPULARITY_WEIGHT * global_pop
        )
        return float(max(0.0, min(1.0, age_adjusted)))

    def _compute_final_score(self, scores: Dict[str, float]) -> float:
        w = self.config
        
        content_combined = (
            scores['watch_history'] * w.watch_history_weight +
            scores['content'] * w.content_weight +
            scores['keyword_overlap'] * w.keyword_overlap_weight
        ) / (w.watch_history_weight + w.content_weight + w.keyword_overlap_weight)
        
        final = (
            scores['imdb'] * w.imdb_weight +
            scores['tfidf'] * w.tfidf_weight +
            content_combined * (1.0 - w.imdb_weight - w.tfidf_weight - w.popularity_weight)
        )
        
        final += scores['popularity'] * w.popularity_weight
        
        penalty = scores.get('negative_penalty', 1.0)
        final = final * penalty
        
        return float(final)

    def get_top_k(self, ranked_movies: List[RankedMovie], k: int = 10) -> List[RankedMovie]:
        return ranked_movies[:k]

    def explain_ranking(self, movie: RankedMovie) -> Dict[str, Any]:
        return {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "final_score": round(movie.score, 4),
            "factor_breakdown": {
                "imdb_rating": round(movie.imdb_score, 4),
                "tfidf_similarity": round(movie.tfidf_score, 4),
                "watch_history_match": round(movie.watch_history_score, 4),
                "semantic_similarity": round(movie.semantic_score, 4),
                "content_similarity": round(movie.content_score, 4),
                "keyword_overlap": round(movie.keyword_overlap_score, 4),
                "popularity": round(movie.popularity_score, 4)
            },
            "weights_used": {
                "imdb": self.config.imdb_weight,
                "tfidf": self.config.tfidf_weight,
                "watch_history": self.config.watch_history_weight,
                "content": self.config.content_weight,
                "keyword_overlap": self.config.keyword_overlap_weight
            }
        }


class DiversityFilter:
    def __init__(self, max_same_genre: int = 5, max_same_director: int = 2):
        self.max_same_genre = max_same_genre
        self.max_same_director = max_same_director

    def filter(self, ranked_movies: List[RankedMovie], feature_matrix) -> List[RankedMovie]:
        selected = []
        genre_counts = {}
        director_counts = {}
        
        for movie in ranked_movies:
            meta = feature_matrix.movie_metadata.get(movie.movie_id, {}) if (feature_matrix and hasattr(feature_matrix, "movie_metadata")) else {}
            genres = movie.genres or meta.get('genres', [])
            directors = meta.get('director', [])
            
            genre_ok = all(genre_counts.get(g, 0) < self.max_same_genre for g in genres) if genres else True
            director_ok = all(director_counts.get(d, 0) < self.max_same_director for d in directors) if directors else True
            
            if genre_ok and director_ok:
                selected.append(movie)
                for g in genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                for d in directors:
                    director_counts[d] = director_counts.get(d, 0) + 1
        
        if len(selected) < min(5, len(ranked_movies)):
            return ranked_movies
        return selected


def main():
    config = RankingConfig()
    ranker = MovieRanker(config)
    print("MovieRanker initialized with config:", config)


if __name__ == "__main__":
    main()
