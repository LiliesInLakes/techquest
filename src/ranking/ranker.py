import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import yaml


@dataclass
class RankingConfig:
    imdb_weight: float = 0.3
    tfidf_weight: float = 0.2
    watch_history_weight: float = 0.5
    semantic_weight: float = 0.3
    content_weight: float = 0.4
    keyword_overlap_weight: float = 0.3
    min_vote_count: int = 100
    popularity_boost: float = 0.1


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

    def _load_config(self, config_path: str) -> RankingConfig:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            rec_cfg = cfg.get("recommendation", {})
            return RankingConfig(
                imdb_weight=rec_cfg.get("imdb_weight", 0.3),
                tfidf_weight=rec_cfg.get("tfidf_weight", 0.2),
                watch_history_weight=rec_cfg.get("watch_history_weight", 0.5),
                semantic_weight=rec_cfg.get("semantic_similarity_threshold", 0.3),
                content_weight=0.4,
                keyword_overlap_weight=0.3
            )
        except FileNotFoundError:
            return RankingConfig()

    def set_dependencies(self, preference_extractor, feature_matrix, semantic_search):
        self.preference_extractor = preference_extractor
        self.feature_matrix = feature_matrix
        self.semantic_search = semantic_search

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
            scores['tfidf'] = self._compute_tfidf_score(movie_id, query_tfidf)
            scores['semantic'] = self._compute_semantic_score(movie_id, query_embedding)
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
        history_boost_factor: float = 0.3
    ) -> List[RankedMovie]:
        """Stage 2: Apply history boost within matching genre clusters"""
        if not stage1_ranked or user_profile is None or self.preference_extractor is None:
            return stage1_ranked
        
        extracted_genres = set(g.lower() for g in extracted_features.genres) if extracted_features else set()
        
        boosted = []
        for rm in stage1_ranked:
            movie_genres = set(g.lower() for g in rm.genres)
            
            genre_match = bool(extracted_genres & movie_genres) if extracted_genres else True
            penalty = self.compute_negative_penalty(rm.movie_id, user_profile, feature_matrix)
            
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

    def compute_negative_penalty(self, movie_id: int, user_profile, feature_matrix=None) -> float:
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
        if fm is None or not hasattr(fm, "movie_metadata") or movie_id not in fm.movie_metadata:
            return 1.0

        meta = fm.movie_metadata[movie_id]
        features = []
        for g in meta.get("genres", []):
            if isinstance(g, str):
                features.append(g.strip().lower())
        for kw in meta.get("keywords", []):
            if isinstance(kw, str):
                features.append(kw.strip().lower())
        for d in meta.get("director", []):
            if isinstance(d, str):
                features.append(d.strip().lower())
        for c in meta.get("cast", [])[:5]:
            if isinstance(c, str):
                features.append(c.strip().lower())

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
        meta = self.feature_matrix.movie_metadata[movie_id]
        
        scores = {}
        
        scores['imdb'] = self._compute_imdb_score(movie)
        scores['tfidf'] = self._compute_tfidf_score(movie_id, query_tfidf)
        scores['watch_history'] = self._compute_watch_history_score(movie_id, user_profile)
        scores['semantic'] = self._compute_semantic_score(movie_id, query_embedding)
        scores['content'] = self._compute_content_score(movie_id, user_profile)
        scores['keyword_overlap'] = self._compute_keyword_overlap_score(movie_id, user_profile)
        scores['popularity'] = self._compute_popularity_score(movie)
        scores['negative_penalty'] = self.compute_negative_penalty(movie_id, user_profile, self.feature_matrix)
        
        return scores

    def _compute_stage1_score(self, scores: Dict[str, float]) -> float:
        w = self.config
        
        final = (
            scores['imdb'] * w.imdb_weight +
            scores['tfidf'] * w.tfidf_weight +
            scores['semantic'] * w.semantic_weight +
            scores['popularity'] * self.config.popularity_boost
        )
        
        remaining_weight = 1.0 - w.imdb_weight - w.tfidf_weight - w.semantic_weight
        if remaining_weight > 0:
            final += scores['popularity'] * remaining_weight
        
        return final

    def _compute_imdb_score(self, movie: Dict[str, Any]) -> float:
        vote_avg = movie.get('vote_average', 0)
        vote_count = movie.get('vote_count', 0)
        
        if vote_count < self.config.min_vote_count:
            return vote_avg / 10.0 * 0.5
        
        return vote_avg / 10.0

    def _compute_tfidf_score(self, movie_id: int, query_tfidf: csr_matrix) -> float:
        if query_tfidf is None or self.feature_matrix is None:
            return 0.0
        
        try:
            movie_idx = self.feature_matrix.movie_ids.index(movie_id)
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

    def _compute_semantic_score(self, movie_id: int, query_embedding: np.ndarray) -> float:
        if query_embedding is None or self.feature_matrix is None:
            return 0.0
        
        try:
            movie_idx = self.feature_matrix.movie_ids.index(movie_id)
            movie_embedding = self.feature_matrix.semantic_embeddings[movie_idx].reshape(1, -1)
            query_emb = query_embedding.reshape(1, -1)
            
            sim = cosine_similarity(query_emb, movie_embedding)[0][0]
            return float(sim)
        except (ValueError, IndexError):
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
        popularity = movie.get('popularity', 0)
        vote_count = movie.get('vote_count', 0)
        
        norm_popularity = min(popularity / 1000.0, 1.0)
        norm_votes = min(vote_count / 10000.0, 1.0)
        
        return (norm_popularity + norm_votes) / 2.0

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
            content_combined * (1 - w.imdb_weight - w.tfidf_weight)
        )
        
        final += scores['popularity'] * self.config.popularity_boost
        
        penalty = scores.get('negative_penalty', 1.0)
        final = final * penalty
        
        return final

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
