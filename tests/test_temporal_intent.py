import pytest
import datetime
from src.llm.nl_feature_extractor import NLFeatureExtractor, ExtractedFeatures
from src.pipeline.recommender import MovieRecommender


class TestTemporalIntent:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = NLFeatureExtractor()
        self.extractor._llm_available = False  # test fallback extraction deterministically
        self.current_year = datetime.date.today().year

    def test_latest_horror_intent(self):
        """Test 'latest horror' extracts narrow recency window and horror genre."""
        features = self.extractor._fallback_extraction("latest horror movie", {})
        assert "Horror" in features.primary_genres
        assert "Horror" in features.explicit_genres
        assert features.release_year_min is not None
        # Must have a recency floor (e.g. current_year - latest_window + 1)
        assert features.release_year_min >= self.current_year - 4
        assert features.release_year_max == self.current_year

    def test_newest_scifi_intent(self):
        """Test 'newest sci-fi' extracts recency window and Science Fiction genre."""
        features = self.extractor._fallback_extraction("newest sci-fi movie", {})
        assert "Science Fiction" in features.primary_genres
        assert features.release_year_min is not None
        assert features.release_year_min >= self.current_year - 4

    def test_recent_thriller_intent(self):
        """Test 'recent thriller' extracts broader recency window and Thriller genre."""
        features = self.extractor._fallback_extraction("recent thriller", {})
        assert "Thriller" in features.primary_genres
        assert features.release_year_min is not None
        assert features.release_year_min == self.current_year - self.extractor.recent_window_years + 1

    def test_this_year_and_last_year_intent(self):
        """Test 'this year' and 'last year' extract exact single-year bounds."""
        f_this = self.extractor._fallback_extraction("great comedy this year", {})
        assert f_this.release_year_min == self.current_year
        assert f_this.release_year_max == self.current_year

        f_last = self.extractor._fallback_extraction("action movie from last year", {})
        assert f_last.release_year_min == self.current_year - 1
        assert f_last.release_year_max == self.current_year - 1

    def test_contextual_new_vs_proper_nouns(self):
        """Test that 'new' as movie recency is extracted, but proper nouns like 'New York' are not."""
        f_new = self.extractor._fallback_extraction("new horror movie", {})
        assert f_new.release_year_min is not None

        f_ny = self.extractor._fallback_extraction("action movie in New York", {})
        assert f_ny.release_year_min is None
        assert f_ny.release_year_max is None

        f_hope = self.extractor._fallback_extraction("movies like A New Hope", {})
        assert f_hope.release_year_min is None
        assert f_hope.release_year_max is None

    def test_explicit_single_year(self):
        """Test '2024 horror movie' extracts exact year bounds."""
        features = self.extractor._fallback_extraction("2024 horror movie", {})
        assert "Horror" in features.primary_genres
        assert features.release_year_min == 2024
        assert features.release_year_max == 2024

    def test_explicit_year_range(self):
        """Test 'movies from 2020 to 2023' extracts inclusive bounds."""
        f1 = self.extractor._fallback_extraction("movies from 2020 to 2023", {})
        assert f1.release_year_min == 2020
        assert f1.release_year_max == 2023

        f2 = self.extractor._fallback_extraction("between 2015 and 2019 drama", {})
        assert f2.release_year_min == 2015
        assert f2.release_year_max == 2019

    def test_strict_and_inclusive_year_operators(self):
        """Test strict operators ('after 2022' -> 2023, 'before 2010' -> 2009)."""
        f_after = self.extractor._fallback_extraction("action movie after 2022", {})
        assert f_after.release_year_min == 2023
        assert f_after.release_year_max is None

        f_before = self.extractor._fallback_extraction("classic comedy before 2010", {})
        assert f_before.release_year_max == 2009
        assert f_before.release_year_min is None

        f_since = self.extractor._fallback_extraction("sci-fi since 2020", {})
        assert f_since.release_year_min == 2020

    def test_query_without_recency(self):
        """Test 'best horror movie' does not set arbitrary recency bounds."""
        features = self.extractor._fallback_extraction("best horror movie", {})
        assert "Horror" in features.primary_genres
        assert features.release_year_min is None
        assert features.release_year_max is None
        assert features.include_adult is None

    def test_temporal_constraint_passed_to_tmdb(self):
        """Test that temporal bounds are forwarded to TMDB /discover/movie parameters."""
        f_latest = self.extractor._fallback_extraction("latest horror movie", {})
        params = self.extractor.build_tmdb_discover_params(f_latest)
        assert "primary_release_date_gte" in params or "primary_release_year" in params
        if "primary_release_date_gte" in params:
            assert params["primary_release_date_gte"].startswith(str(f_latest.release_year_min))

        f_2024 = self.extractor._fallback_extraction("2024 action movie", {})
        params_2024 = self.extractor.build_tmdb_discover_params(f_2024)
        assert params_2024.get("primary_release_year") == 2024

    def test_candidate_filter_strictly_excludes_old_movies(self):
        """Verify that 1970s/1980s movies like Jaws (1975) cannot survive candidate filtering when recency is requested."""
        recommender = MovieRecommender()
        f_latest = self.extractor._fallback_extraction("latest horror movie", {})

        old_movie = {
            'id': 578,
            'title': 'Jaws (1975)',
            'genres': ['Horror', 'Adventure'],
            'release_date': '1975-06-20',
            'vote_average': 8.0,
            'vote_count': 10000
        }
        recent_movie = {
            'id': 999999,
            'title': 'Recent Horror (2023)',
            'genres': ['Horror'],
            'release_date': '2023-10-15',
            'vote_average': 7.5,
            'vote_count': 500
        }

        assert recommender._passes_all_hard_filters(old_movie, f_latest) is False
        assert recommender._passes_all_hard_filters(recent_movie, f_latest) is True

    def test_barbarian_2022_rejected_and_m3gan_2023_allowed(self):
        """Focused regression test: 'latest horror movie' must reject 2022 movies (e.g. Barbarian)
        and allow 2023 movies (e.g. M3GAN) when latest window starts at 2023."""
        recommender = MovieRecommender()
        features = self.extractor._fallback_extraction("latest horror movie", {})

        assert features.release_year_min == 2023
        assert features.release_year_max == self.current_year

        barbarian_2022 = {
            'id': 913290,
            'title': 'Barbarian (2022)',
            'genres': ['Horror', 'Thriller'],
            'release_date': '2022-09-08',
            'vote_average': 7.03,
            'vote_count': 730
        }
        m3gan_2023 = {
            'id': 536554,
            'title': 'M3GAN (2023)',
            'genres': ['Horror', 'Science Fiction'],
            'release_date': '2023-01-06',
            'vote_average': 7.0,
            'vote_count': 3000
        }

        # 2022 movie MUST be rejected under 'latest'
        assert recommender._passes_all_hard_filters(barbarian_2022, features) is False
        # 2023 movie MUST be allowed under 'latest'
        assert recommender._passes_all_hard_filters(m3gan_2023, features) is True

        # For 'recent horror movie' (broader window), 2022 movie is allowed
        f_recent = self.extractor._fallback_extraction("recent horror movie", {})
        assert recommender._passes_all_hard_filters(barbarian_2022, f_recent) is True
        assert recommender._passes_all_hard_filters(m3gan_2023, f_recent) is True

    def test_llm_parse_and_validate_temporal_refinement(self):
        """Test that _parse_and_validate deterministically refines LLM output to respect configured temporal intent."""
        # Simulated LLM output that incorrectly returned a broader window (e.g. 2021)
        llm_dict = {"primary_genres": ["Horror"], "release_year_min": 2021, "release_year_max": 2026, "semantic_search_query": "latest horror"}
        validated = self.extractor._parse_and_validate(llm_dict, "latest horror movie")

        assert validated.release_year_min == 2023
        assert validated.release_year_max == 2026