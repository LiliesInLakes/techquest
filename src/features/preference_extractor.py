import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math


@dataclass
class UserPreferenceProfile:
    user_id: str
    top_genres: List[Tuple[str, float]]
    top_keywords: List[Tuple[str, float]]
    top_directors: List[Tuple[str, float]]
    top_cast: List[Tuple[str, float]]
    rating_distribution: Dict[str, float]
    avg_rating: float
    rating_std: float
    preferred_decades: List[Tuple[str, float]]
    preferred_languages: List[Tuple[str, float]]
    watch_count: int
    feature_vector: np.ndarray
    keyword_weights: Dict[str, float]
    genre_weights: Dict[str, float]
    top_tags: List[Tuple[str, float]] = field(default_factory=list)
    top_themes: List[Tuple[str, float]] = field(default_factory=list)
    top_tones: List[Tuple[str, float]] = field(default_factory=list)
    negative_preferences: Dict[str, float] = field(default_factory=dict)


class PreferenceExtractor:
    def __init__(self, feature_matrix=None, semantic_search=None):
        self.feature_matrix = feature_matrix
        self.semantic_search = semantic_search

    def extract_preferences(
        self,
        watch_history: List[Dict[str, Any]],
        ratings: Dict[int, float] = None,
        feature_matrix=None
    ) -> UserPreferenceProfile:
        if not watch_history:
            return self._empty_profile("unknown")
        
        ratings = ratings or {}
        
        genre_counter = Counter()
        keyword_counter = Counter()
        director_counter = Counter()
        cast_counter = Counter()
        decade_counter = Counter()
        language_counter = Counter()
        
        total_rating = 0.0
        rating_count = 0
        rating_values = []
        
        movie_features = []
        
        for entry in watch_history:
            movie_id = entry.get('movie_id') or entry.get('id')
            user_rating = ratings.get(movie_id, entry.get('rating', 0))
            
            if user_rating > 0:
                rating_values.append(user_rating)
                total_rating += user_rating
                rating_count += 1
            
            movie_data = self._get_movie_data(movie_id, feature_matrix)
            if not movie_data:
                continue
            
            weight = self._compute_weight(user_rating)
            
            for genre in movie_data.get('genres', []):
                genre_counter[genre] += weight
            
            for keyword in movie_data.get('keywords', []):
                keyword_counter[keyword] += weight
            
            for director in movie_data.get('director', []):
                director_counter[director] += weight
            
            for actor in movie_data.get('cast', [])[:5]:
                cast_counter[actor] += weight
            
            release_date = movie_data.get('release_date', '')
            if release_date:
                try:
                    year = int(release_date[:4])
                    decade = f"{(year // 10) * 10}s"
                    decade_counter[decade] += weight
                except:
                    pass
            
            lang = movie_data.get('original_language', 'en')
            language_counter[lang] += weight
            
            if feature_matrix and movie_id in feature_matrix.movie_metadata:
                idx = feature_matrix.movie_ids.index(movie_id)
                movie_features.append((feature_matrix.count_matrix[idx], weight))
        
        feature_vector = self._compute_feature_vector(movie_features, feature_matrix)
        
        keyword_weights = self._compute_keyword_weights(keyword_counter, watch_history, ratings)
        genre_weights = self._compute_genre_weights(genre_counter, watch_history, ratings)
        
        return UserPreferenceProfile(
            user_id="user",
            top_genres=self._normalize_and_sort(genre_counter, top_n=10),
            top_keywords=self._normalize_and_sort(keyword_counter, top_n=20),
            top_directors=self._normalize_and_sort(director_counter, top_n=10),
            top_cast=self._normalize_and_sort(cast_counter, top_n=10),
            rating_distribution=self._compute_rating_distribution(rating_values),
            avg_rating=total_rating / rating_count if rating_count > 0 else 0,
            rating_std=np.std(rating_values) if rating_values else 0,
            preferred_decades=self._normalize_and_sort(decade_counter, top_n=5),
            preferred_languages=self._normalize_and_sort(language_counter, top_n=5),
            watch_count=len(watch_history),
            feature_vector=feature_vector,
            keyword_weights=keyword_weights,
            genre_weights=genre_weights
        )

    def _get_movie_data(self, movie_id: int, feature_matrix) -> Optional[Dict]:
        if feature_matrix and movie_id in feature_matrix.movie_metadata:
            return feature_matrix.movie_metadata[movie_id]
        return None

    def _compute_weight(self, user_rating: float) -> float:
        if user_rating <= 0:
            return 0.5
        return 0.3 + (user_rating / 10.0) * 0.7

    def _compute_feature_vector(self, movie_features: List[Tuple], feature_matrix) -> np.ndarray:
        if not movie_features or feature_matrix is None:
            return np.zeros(feature_matrix.count_matrix.shape[1] if feature_matrix else 1000)
        
        weighted_sum = np.zeros(feature_matrix.count_matrix.shape[1])
        total_weight = 0
        
        for features, weight in movie_features:
            if hasattr(features, 'toarray'):
                features = features.toarray().flatten()
            weighted_sum += features * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum /= total_weight
        
        return weighted_sum

    def _compute_keyword_weights(self, counter: Counter, watch_history: List[Dict], ratings: Dict) -> Dict[str, float]:
        total_watched = len(watch_history)
        if total_watched == 0:
            return {}
        
        weights = {}
        for keyword, count in counter.items():
            tf = count / total_watched
            
            idf = 1.0
            if self.feature_matrix:
                doc_freq = sum(1 for meta in self.feature_matrix.movie_metadata.values() 
                              if keyword in meta.get('keywords', []))
                if doc_freq > 0:
                    idf = math.log(len(self.feature_matrix.movie_metadata) / doc_freq)
            
            weights[keyword] = tf * idf
        
        return weights

    def _compute_genre_weights(self, counter: Counter, watch_history: List[Dict], ratings: Dict) -> Dict[str, float]:
        total_watched = len(watch_history)
        if total_watched == 0:
            return {}
        
        weights = {}
        for genre, count in counter.items():
            tf = count / total_watched
            
            idf = 1.0
            if self.feature_matrix:
                doc_freq = sum(1 for meta in self.feature_matrix.movie_metadata.values() 
                              if genre in meta.get('genres', []))
                if doc_freq > 0:
                    idf = math.log(len(self.feature_matrix.movie_metadata) / doc_freq)
            
            weights[genre] = tf * idf
        
        return weights

    def _normalize_and_sort(self, counter: Counter, top_n: int = 10) -> List[Tuple[str, float]]:
        if not counter:
            return []
        
        total = sum(counter.values())
        if total == 0:
            return []
        
        normalized = [(k, v / total) for k, v in counter.items()]
        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized[:top_n]

    def _compute_rating_distribution(self, ratings: List[float]) -> Dict[str, float]:
        if not ratings:
            return {}
        
        bins = {
            "0.5-2.0": 0, "2.5-4.0": 0, "4.5-6.0": 0,
            "6.5-7.5": 0, "8.0-10.0": 0
        }
        
        for r in ratings:
            if r <= 2.0:
                bins["0.5-2.0"] += 1
            elif r <= 4.0:
                bins["2.5-4.0"] += 1
            elif r <= 6.0:
                bins["4.5-6.0"] += 1
            elif r <= 7.5:
                bins["6.5-7.5"] += 1
            else:
                bins["8.0-10.0"] += 1
        
        total = len(ratings)
        return {k: v / total for k, v in bins.items()}

    def _empty_profile(self, user_id: str) -> UserPreferenceProfile:
        return UserPreferenceProfile(
            user_id=user_id,
            top_genres=[],
            top_keywords=[],
            top_directors=[],
            top_cast=[],
            rating_distribution={},
            avg_rating=0,
            rating_std=0,
            preferred_decades=[],
            preferred_languages=[],
            watch_count=0,
            feature_vector=np.zeros(1000),
            keyword_weights={},
            genre_weights={}
        )

    def compute_content_similarity(self, profile: UserPreferenceProfile, movie_id: int, feature_matrix) -> float:
        if movie_id not in feature_matrix.movie_metadata:
            return 0.0
        
        movie_idx = feature_matrix.movie_ids.index(movie_id)
        movie_vector = feature_matrix.count_matrix[movie_idx]
        if hasattr(movie_vector, 'toarray'):
            movie_vector = movie_vector.toarray().flatten()
        
        if np.linalg.norm(profile.feature_vector) == 0 or np.linalg.norm(movie_vector) == 0:
            return 0.0
        
        similarity = np.dot(profile.feature_vector, movie_vector) / (
            np.linalg.norm(profile.feature_vector) * np.linalg.norm(movie_vector)
        )
        
        return float(similarity)

    def compute_keyword_overlap_score(self, profile: UserPreferenceProfile, movie_id: int, feature_matrix) -> float:
        if movie_id not in feature_matrix.movie_metadata:
            return 0.0
        
        movie_meta = feature_matrix.movie_metadata[movie_id]
        movie_keywords = set(movie_meta.get('keywords', []))
        movie_genres = set(movie_meta.get('genres', []))
        
        if not movie_keywords and not movie_genres:
            return 0.0
        
        user_keywords = set(kw for kw, _ in profile.top_keywords)
        user_genres = set(g for g, _ in profile.top_genres)
        
        keyword_overlap = len(user_keywords & movie_keywords) / max(len(user_keywords), 1)
        genre_overlap = len(user_genres & movie_genres) / max(len(user_genres), 1)
        
        weighted_keyword_score = sum(
            profile.keyword_weights.get(kw, 0) for kw in user_keywords & movie_keywords
        )
        weighted_genre_score = sum(
            profile.genre_weights.get(g, 0) for g in user_genres & movie_genres
        )
        
        return (keyword_overlap * 0.4 + genre_overlap * 0.3 + 
                weighted_keyword_score * 0.2 + weighted_genre_score * 0.1)


def main():
    extractor = PreferenceExtractor()
    
    sample_history = [
        {"movie_id": 550, "rating": 9.0},
        {"movie_id": 680, "rating": 8.5},
        {"movie_id": 13, "rating": 8.0},
        {"movie_id": 278, "rating": 9.5},
    ]
    
    profile = extractor.extract_preferences(sample_history)
    print(f"Top genres: {profile.top_genres}")
    print(f"Top keywords: {profile.top_keywords[:10]}")
    print(f"Avg rating: {profile.avg_rating}")


if __name__ == "__main__":
    main()