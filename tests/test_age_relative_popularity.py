import pytest
import math
import numpy as np
from src.ranking.ranker import (
    MovieRanker,
    RankingConfig,
    RankedMovie,
    RATING_WEIGHT,
    TFIDF_WEIGHT,
    SEMANTIC_WEIGHT,
    POPULARITY_WEIGHT,
    PRIMARY_COHORT_RADIUS,
    SECONDARY_COHORT_RADIUS,
    COHORT_POPULARITY_WEIGHT,
    GLOBAL_POPULARITY_WEIGHT,
    MIN_COHORT_SIZE
)


class MockFeatureMatrix:
    def __init__(self, movie_metadata=None):
        self.movie_metadata = movie_metadata or {}


class TestAgeRelativePopularity:
    def setup_method(self):
        # Create a synthetic dataset with distinct cohorts:
        # 1999 cohort (30 movies, counts 100 to 50,000)
        # 2022 cohort (30 movies, counts 10 to 2,000)
        # 1930 cohort (sparse: only 3 movies)
        metadata = {}
        mid = 1
        for count in np.linspace(100, 50000, 30):
            metadata[mid] = {
                'id': mid,
                'title': f'Classic Film 1999 #{mid} (1999)',
                'release_date': '1999-06-15',
                'rating_count': int(count),
                'vote_average': 8.0
            }
            mid += 1

        for count in np.linspace(10, 2000, 30):
            metadata[mid] = {
                'id': mid,
                'title': f'Modern Film 2022 #{mid}',
                'release_date': '2022-11-20',
                'rating_count': int(count),
                'vote_average': 7.5
            }
            mid += 1

        for count in [5, 10, 15]:
            metadata[mid] = {
                'id': mid,
                'title': f'Rare Film 1930 #{mid} (1930)',
                'release_date': '1930-01-01',
                'rating_count': int(count),
                'vote_average': 6.5
            }
            mid += 1

        self.mock_fm = MockFeatureMatrix(metadata)
        self.ranker = MovieRanker(RankingConfig())
        self.ranker.set_dependencies(None, self.mock_fm, None)

    def test_exact_stage1_weights_and_sum_to_one(self):
        """Verify exact weights (0.30, 0.25, 0.35, 0.10) and sum == 1.00."""
        cfg = self.ranker.config
        assert cfg.imdb_weight == pytest.approx(0.30)
        assert cfg.tfidf_weight == pytest.approx(0.25)
        assert cfg.semantic_weight == pytest.approx(0.35)
        assert cfg.popularity_weight == pytest.approx(0.10)

        total_weight = (
            cfg.imdb_weight +
            cfg.tfidf_weight +
            cfg.semantic_weight +
            cfg.popularity_weight
        )
        assert total_weight == pytest.approx(1.00)
        assert RATING_WEIGHT == 0.30
        assert TFIDF_WEIGHT == 0.25
        assert SEMANTIC_WEIGHT == 0.35
        assert POPULARITY_WEIGHT == 0.10

    def test_popularity_contributes_only_10_percent_and_no_double_counting(self):
        """Verify Stage 1 formula is strictly linear with exactly 10% popularity weight."""
        scores = {
            'imdb': 1.0,
            'tfidf': 0.0,
            'semantic': 0.0,
            'popularity': 0.0
        }
        assert self.ranker._compute_stage1_score(scores) == pytest.approx(0.30)

        scores['popularity'] = 1.0
        # If popularity weight is exactly 0.10, score should increase by exactly 0.10
        assert self.ranker._compute_stage1_score(scores) == pytest.approx(0.40)

        # Full score when all components are 1.0 must be 1.00
        all_ones = {'imdb': 1.0, 'tfidf': 1.0, 'semantic': 1.0, 'popularity': 1.0}
        assert self.ranker._compute_stage1_score(all_ones) == pytest.approx(1.00)

    def test_zero_and_null_rating_count_handling(self):
        """Verify zero, negative, or NULL rating_count deterministically return 0.0."""
        movie_zero = {'title': 'Zero Votes (2022)', 'rating_count': 0, 'release_date': '2022-01-01'}
        movie_null = {'title': 'Null Votes (2022)', 'rating_count': None, 'release_date': '2022-01-01'}
        movie_empty = {'title': 'No Votes Field (2022)', 'release_date': '2022-01-01'}
        movie_neg = {'title': 'Negative Votes (2022)', 'rating_count': -5, 'release_date': '2022-01-01'}

        assert self.ranker._compute_popularity_score(movie_zero) == 0.0
        assert self.ranker._compute_popularity_score(movie_null) == 0.0
        assert self.ranker._compute_popularity_score(movie_empty) == 0.0
        assert self.ranker._compute_popularity_score(movie_neg) == 0.0

    def test_missing_release_year_falls_back_to_global_popularity(self):
        """Verify missing release year falls back to pure global log popularity."""
        movie_no_year = {'title': 'Unknown Year Movie', 'rating_count': 5000}
        score = self.ranker._compute_popularity_score(movie_no_year)
        
        # Expected global popularity: log1p(5000) / log1p(max)
        expected_global = math.log1p(5000) / self.ranker._global_log_max
        assert score == pytest.approx(expected_global, rel=1e-4)

    def test_sparse_cohort_fallback(self):
        """Verify sparse cohorts (< MIN_COHORT_SIZE) fall back to global popularity."""
        # 1930 cohort only has 3 movies in setup (< 20 items)
        movie_1930 = {'title': 'Sparse 1930 (1930)', 'release_date': '1930-05-01', 'rating_count': 100}
        score = self.ranker._compute_popularity_score(movie_1930)
        
        expected_global = math.log1p(100) / self.ranker._global_log_max
        assert score == pytest.approx(expected_global, rel=1e-4)

    def test_cohort_percentile_and_age_adjusted_popularity(self):
        """Verify top movie in a modern cohort achieves high percentile relative to peers."""
        # Top 2022 film with 2,000 votes in a cohort where max is 2,000
        top_2022 = {'title': 'Top 2022 Film', 'release_date': '2022-12-01', 'rating_count': 2000}
        score_2022 = self.ranker._compute_popularity_score(top_2022)

        # Top 1999 film with 50,000 votes in a cohort where max is 50,000
        top_1999 = {'title': 'Top 1999 Film (1999)', 'release_date': '1999-07-01', 'rating_count': 50000}
        score_1999 = self.ranker._compute_popularity_score(top_1999)

        # Both are top of their respective release cohorts (percentile ~ 1.0)
        # Even though 1999 has 25x more votes (50k vs 2k), the age-adjusted popularity difference is modest
        assert score_2022 >= 0.85
        assert score_1999 >= 0.95
        # Popularity contribution gap in Stage 1 is at most (score_1999 - score_2022) * 0.10
        gap_in_stage1 = (score_1999 - score_2022) * POPULARITY_WEIGHT
        assert gap_in_stage1 < 0.02, f"Expected Stage 1 popularity gap < 0.02, got {gap_in_stage1}"

    def test_release_year_extraction_robustness(self):
        """Verify release year extraction from release_date, release_year, and embedded title."""
        assert self.ranker._extract_release_year({'release_date': '1994-09-23'}) == 1994
        assert self.ranker._extract_release_year({'release_year': 2010}) == 2010
        assert self.ranker._extract_release_year({'year': '2015'}) == 2015
        assert self.ranker._extract_release_year({'title': 'Pulp Fiction (1994)'}) == 1994
        assert self.ranker._extract_release_year({'title': 'Movie Without Year'}) is None
