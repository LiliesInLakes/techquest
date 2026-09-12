import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from src.features.preference_extractor import PreferenceExtractor, UserPreferenceProfile
from src.ranking.ranker import MovieRanker, RankingConfig, RankedMovie, DiversityFilter
from src.features.semantic_search import SemanticKeywordSearch, HybridKeywordMatcher
from src.llm.reasoning import DeepSeekReasoner, ReasoningTemplateEngine, RecommendationReasoning
from src.pipeline.recommender import MovieRecommender, RecommendationResult


class TestPreferenceExtractor:
    def test_extract_preferences_empty_history(self):
        extractor = PreferenceExtractor()
        profile = extractor.extract_preferences([])
        assert profile.watch_count == 0
        assert profile.avg_rating == 0

    def test_extract_preferences_with_history(self):
        extractor = PreferenceExtractor()
        watch_history = [
            {"movie_id": 1, "rating": 9.0},
            {"movie_id": 2, "rating": 8.0},
            {"movie_id": 3, "rating": 7.0},
        ]
        ratings = {1: 9.0, 2: 8.0, 3: 7.0}
        
        mock_matrix = Mock()
        mock_matrix.movie_metadata = {
            1: {'genres': ['Action', 'Sci-Fi'], 'keywords': ['space', 'future'], 'director': ['Director A'], 'cast': ['Actor X'], 'release_date': '2020-01-01', 'original_language': 'en'},
            2: {'genres': ['Action', 'Thriller'], 'keywords': ['chase', 'explosion'], 'director': ['Director B'], 'cast': ['Actor Y'], 'release_date': '2019-01-01', 'original_language': 'en'},
            3: {'genres': ['Drama'], 'keywords': ['emotional', 'life'], 'director': ['Director C'], 'cast': ['Actor Z'], 'release_date': '2021-01-01', 'original_language': 'en'},
        }
        mock_matrix.count_matrix = np.random.rand(3, 100)
        mock_matrix.movie_ids = [1, 2, 3]
        
        extractor.feature_matrix = mock_matrix
        profile = extractor.extract_preferences(watch_history, ratings, mock_matrix)
        
        assert profile.watch_count == 3
        assert profile.avg_rating == 8.0
        assert len(profile.top_genres) > 0
        assert 'Action' in [g for g, _ in profile.top_genres]

    def test_compute_weight(self):
        extractor = PreferenceExtractor()
        assert extractor._compute_weight(0) == 0.5
        assert extractor._compute_weight(5.0) == pytest.approx(0.65)
        assert extractor._compute_weight(10.0) == pytest.approx(1.0)


class TestMovieRanker:
    def setup_method(self):
        self.config = RankingConfig(
            imdb_weight=0.3,
            tfidf_weight=0.2,
            watch_history_weight=0.5
        )
        self.ranker = MovieRanker(self.config)

    def test_compute_imdb_score(self):
        movie_high = {'vote_average': 8.5, 'vote_count': 1000}
        movie_low = {'vote_average': 5.0, 'vote_count': 50}
        movie_no_votes = {'vote_average': 9.0, 'vote_count': 10}
        
        assert self.ranker._compute_imdb_score(movie_high) == 0.85
        # 50 votes < default min_vote_count (100) -> score halved: 5.0/10*0.5
        assert self.ranker._compute_imdb_score(movie_low) == pytest.approx(0.25)
        assert self.ranker._compute_imdb_score(movie_no_votes) == pytest.approx(0.45)

    def test_compute_popularity_score(self):
        movie_popular = {'popularity': 5000, 'vote_count': 20000}
        movie_average = {'popularity': 100, 'vote_count': 1000}
        
        score_pop = self.ranker._compute_popularity_score(movie_popular)
        score_avg = self.ranker._compute_popularity_score(movie_average)
        
        assert 0 <= score_pop <= 1
        assert 0 <= score_avg <= 1
        assert score_pop > score_avg


class TestSemanticKeywordSearch:
    def test_build_index(self):
        search = SemanticKeywordSearch(threshold=0.5)
        keywords = ["action", "comedy", "drama", "sci-fi", "thriller"]
        search.build_index(keywords)
        
        assert len(search.keyword_list) == 5
        assert search.embedding_matrix is not None
        assert search.embedding_matrix.shape == (5, 384)

    def test_find_similar_keywords(self):
        search = SemanticKeywordSearch(threshold=0.5)
        keywords = ["action", "comedy", "drama", "sci-fi", "thriller", "adventure"]
        search.build_index(keywords)
        
        similar = search.find_similar_keywords("action movie", top_k=3)
        assert len(similar) <= 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)

    def test_expand_query_with_synonyms(self):
        search = SemanticKeywordSearch(threshold=0.5)
        keywords = ["funny", "humor", "comedy", "laugh", "hilarious"]
        search.build_index(keywords)
        
        expanded = search.expand_query_with_synonyms("funny movie")
        assert "funny" in expanded
        assert len(expanded) >= 1


class TestReasoningTemplateEngine:
    def test_generate_genre_match(self):
        engine = ReasoningTemplateEngine()
        movie = {'title': 'Test Movie', 'genres': ['Action', 'Sci-Fi'], 'vote_average': 8.0}
        profile = Mock()
        profile.top_genres = [('Action', 0.5), ('Sci-Fi', 0.3)]
        profile.avg_rating = 7.5
        factors = {'watch_history': 0.3, 'imdb': 0.8}
        
        reasoning = engine.generate(movie, profile, factors)
        assert 'Test Movie' in reasoning
        assert 'Action' in reasoning

    def test_generate_high_rating(self):
        engine = ReasoningTemplateEngine()
        movie = {'title': 'Great Movie', 'genres': ['Drama'], 'vote_average': 9.0}
        profile = Mock()
        profile.top_genres = []
        profile.avg_rating = 8.0
        factors = {'watch_history': 0.2, 'imdb': 0.9}
        
        reasoning = engine.generate(movie, profile, factors)
        assert 'Great Movie' in reasoning
        assert '8.0' in reasoning


class TestDiversityFilter:
    def test_filter_diversity(self):
        filter = DiversityFilter(max_same_genre=2, max_same_director=1)
        
        mock_matrix = Mock()
        mock_matrix.movie_metadata = {
            1: {'genres': ['Action'], 'director': ['Director A']},
            2: {'genres': ['Action'], 'director': ['Director B']},
            3: {'genres': ['Action'], 'director': ['Director C']},
            4: {'genres': ['Comedy'], 'director': ['Director D']},
            5: {'genres': ['Drama'], 'director': ['Director A']},
        }
        
        movies = [
            RankedMovie(movie_id=i, title=f"Movie {i}", score=1.0-i*0.1,
                       imdb_score=0, tfidf_score=0, watch_history_score=0,
                       semantic_score=0, content_score=0, keyword_overlap_score=0,
                       popularity_score=0, overview='', genres=['Action'],
                       vote_average=8.0, vote_count=1000, reasoning_factors={})
            for i in range(1, 6)
        ]
        
        filtered = filter.filter(movies, mock_matrix)
        assert len(filtered) <= 5


class TestMovieRecommender:
    @pytest.mark.asyncio
    async def test_build_query_from_profile(self):
        recommender = MovieRecommender()
        
        profile = Mock()
        profile.top_genres = [('Action', 0.5), ('Sci-Fi', 0.3)]
        profile.top_keywords = [('space', 0.4), ('future', 0.3)]
        
        query = recommender._build_query_from_profile(profile, {'query': 'exciting', 'mood': 'adventure'})
        
        assert 'Action' in query
        assert 'Sci-Fi' in query
        assert 'space' in query
        assert 'exciting' in query
        assert 'adventure' in query

    @pytest.mark.asyncio
    async def test_summarize_watch_history(self):
        recommender = MovieRecommender()
        
        history = [
            {"movie_id": 1, "rating": 9.0},
            {"movie_id": 2, "rating": 8.0},
            {"movie_id": 3, "rating": 0},
        ]
        
        summary = recommender._summarize_watch_history(history)
        assert "3 movies" in summary
        assert "2 of them" in summary
        assert "8.5" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])