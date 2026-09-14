import os
import time
import math
import pytest
import numpy as np

from src.memory.models import (
    EventType, FeatureType, WatchEvent, RatingEvent,
    FeaturePreference, NegativePreference, UserProfileMemory
)
from src.memory.decay import compute_exponential_decay, aggregate_decayed_signals
from src.memory.storage import MemoryStorage
from src.memory.engine import (
    UserMemoryEngine,
    EVENT_WEIGHTS,
    map_rating_to_signals,
    compute_beta,
    compute_confidence,
    compute_effective_binge_weight,
    asymptotic_addition_update
)
from src.memory.service import UserMemoryService
from src.ranking.ranker import MovieRanker, RankingConfig, RankedMovie


class TestAsymptoticAddition:
    def test_asymptotic_addition_math(self):
        # Initial score 0.80, alpha 0.10, signal 0.50
        # Expected: 0.80 + 0.10 * 0.50 * (1.0 - 0.80) = 0.80 + 0.01 = 0.81
        updated = asymptotic_addition_update(0.80, 0.50, alpha=0.10)
        assert pytest.approx(updated, 1e-5) == 0.81

    def test_asymptotic_strictly_increasing(self):
        # Starting from 0.0, positive signals strictly increase score
        score = 0.0
        for _ in range(5):
            new_score = asymptotic_addition_update(score, 0.10, alpha=0.10)
            assert new_score > score
            score = new_score
        assert score <= 1.0

    def test_asymptotic_upper_bound(self):
        # Starting from 0.99 with large signal, cannot exceed 1.0
        score = asymptotic_addition_update(0.99, 10.0, alpha=0.10)
        assert score <= 1.0
        assert score >= 0.99


class TestBetaTransition:
    def test_smooth_beta_curve(self):
        # beta(N) = 0.40 + 0.60 * max(0, 1 - N / 10)
        assert pytest.approx(compute_beta(0), 1e-5) == 1.00
        assert pytest.approx(compute_beta(3), 1e-5) == 0.82
        assert pytest.approx(compute_beta(5), 1e-5) == 0.70
        assert pytest.approx(compute_beta(10), 1e-5) == 0.40
        assert pytest.approx(compute_beta(20), 1e-5) == 0.40


class TestBlendedNegativePenalty:
    def test_blended_negative_penalty_math(self):
        # penalty = 1.0 - min(0.9, 0.75 * max + 0.25 * avg)
        # 4 features with negative scores [1.0, 0.0, 0.0, 0.0]
        # max = 1.0, avg = 0.25
        # raw_penalty = 0.75 * 1.0 + 0.25 * 0.25 = 0.8125
        # penalty = 1.0 - 0.8125 = 0.1875
        ranker = MovieRanker()
        class DummyProfile:
            negative_preferences = {"action": 1.0}

        class DummyFeatureMatrix:
            movie_metadata = {
                101: {
                    "genres": ["Action", "Comedy", "Drama", "Sci-Fi"]
                }
            }

        penalty = ranker.compute_negative_penalty(101, DummyProfile(), DummyFeatureMatrix())
        assert pytest.approx(penalty, 1e-4) == 0.1875

    def test_no_negative_features_penalty_is_one(self):
        ranker = MovieRanker()
        class DummyProfile:
            negative_preferences = {"horror": 1.0}

        class DummyFeatureMatrix:
            movie_metadata = {
                102: {
                    "genres": ["Comedy", "Romance"]
                }
            }

        penalty = ranker.compute_negative_penalty(102, DummyProfile(), DummyFeatureMatrix())
        assert penalty == 1.0

    def test_empty_profile_penalty_is_one(self):
        ranker = MovieRanker()
        penalty = ranker.compute_negative_penalty(102, None, None)
        assert penalty == 1.0


class TestRatingSpectrum:
    def test_rating_5_stars(self):
        is_pos, pos_w, is_neg, neg_p, is_neu = map_rating_to_signals(5.0)
        assert is_pos is True
        assert pos_w == 1.0
        assert is_neg is False
        assert is_neu is False

    def test_rating_4_stars(self):
        is_pos, pos_w, is_neg, neg_p, is_neu = map_rating_to_signals(4.0)
        assert is_pos is True
        assert pos_w == 0.6
        assert is_neg is False
        assert is_neu is False

    def test_rating_3_stars_neutral(self):
        is_pos, pos_w, is_neg, neg_p, is_neu = map_rating_to_signals(3.0)
        assert is_pos is False
        assert pos_w == 0.0
        assert is_neg is False
        assert neg_p == 0.0
        assert is_neu is True

    def test_rating_2_stars_mild_negative(self):
        is_pos, pos_w, is_neg, neg_p, is_neu = map_rating_to_signals(2.0)
        assert is_pos is False
        assert is_neg is True
        assert neg_p == 0.4
        assert is_neu is False

    def test_rating_1_and_0_stars_strong_negative(self):
        for r in [1.0, 0.0]:
            is_pos, pos_w, is_neg, neg_p, is_neu = map_rating_to_signals(r)
            assert is_pos is False
            assert is_neg is True
            assert neg_p == 1.0
            assert is_neu is False


class TestTimeDecayAndBingeDamping:
    def test_half_life_decay_at_known_intervals(self):
        # Half life H = 7 days
        # t = 0 -> 1.0
        assert pytest.approx(compute_exponential_decay(0.0, 7.0), 1e-5) == 1.0
        # t = 7 -> 0.50
        assert pytest.approx(compute_exponential_decay(7.0, 7.0), 1e-5) == 0.50
        # t = 14 -> 0.25
        assert pytest.approx(compute_exponential_decay(14.0, 7.0), 1e-5) == 0.25
        # t = 21 -> 0.125
        assert pytest.approx(compute_exponential_decay(21.0, 7.0), 1e-5) == 0.125

    def test_logarithmic_binge_damping_math(self):
        # sum weight = 5.0 -> log2(1 + 5) = log2(6) ~= 2.58496
        assert pytest.approx(compute_effective_binge_weight(5.0), 1e-4) == math.log2(6.0)
        assert compute_effective_binge_weight(0.0) == 0.0

    def test_bayesian_confidence_scaling(self):
        # confidence = N / (N + 10)
        assert compute_confidence(0, 10.0) == 0.0
        assert pytest.approx(compute_confidence(10, 10.0), 1e-5) == 0.50
        assert pytest.approx(compute_confidence(90, 10.0), 1e-5) == 0.90


class TestMemoryStorageAndService:
    @pytest.fixture
    def test_db_path(self, tmp_path):
        return str(tmp_path / "test_memory.db")

    def test_end_to_end_event_flow(self, test_db_path):
        service = UserMemoryService(db_path=test_db_path)
        user_id = "test_user_1"

        meta = {
            "genres": ["Sci-Fi", "Thriller"],
            "keywords": ["space", "artificial intelligence"],
            "director": ["Christopher Nolan"],
            "cast": ["Matthew McConaughey", "Anne Hathaway"],
            "release_date": "2014-11-07"
        }

        # 1. Log movie_clicked
        ev1 = service.record_interaction(user_id, 101, EventType.MOVIE_CLICKED, metadata=meta)
        assert ev1.id is not None
        assert ev1.signal_weight == 0.10

        # Check profile
        prof = service.get_user_profile(user_id)
        assert prof.total_interactions == 1
        assert "genre" in prof.long_term_preferences
        assert "sci-fi" in prof.long_term_preferences["genre"]
        sci_fi_pref = prof.long_term_preferences["genre"]["sci-fi"]
        assert sci_fi_pref.interaction_count == 1
        assert sci_fi_pref.score > 0.0

        # 2. Rate neutral 3 stars
        r = service.record_rating(user_id, 101, rating=3.0)
        assert r.id is not None
        prof = service.get_user_profile(user_id)
        # Total interactions incremented
        assert prof.total_interactions == 2
        # Sci-Fi count incremented but score did not artificially drop or jump
        sci_fi_pref2 = prof.long_term_preferences["genre"]["sci-fi"]
        assert sci_fi_pref2.interaction_count == 2
        assert len(prof.negative_preferences) == 0  # No negative preferences from 3 stars

        # 3. Rate a horror movie 1 star (explicit negative)
        horror_meta = {"genres": ["Horror"]}
        service.record_interaction(user_id, 202, EventType.MOVIE_VIEWED, metadata=horror_meta)
        service.record_rating(user_id, 202, rating=1.0)

        prof = service.get_user_profile(user_id)
        assert "genre" in prof.negative_preferences
        assert "horror" in prof.negative_preferences["genre"]
        assert prof.negative_preferences["genre"]["horror"].penalty_weight > 0.0

        # 4. Adapter conversion
        adapter = service.get_combined_preference_profile(user_id)
        assert hasattr(adapter, "negative_preferences")
        assert "horror" in adapter.negative_preferences

    def test_profile_replay_idempotence(self, test_db_path):
        service = UserMemoryService(db_path=test_db_path)
        user_id = "replay_user"

        meta = {"genres": ["Animation", "Family"]}
        service.record_interaction(user_id, 301, EventType.MARKED_WATCHED, metadata=meta)
        service.record_rating(user_id, 301, rating=5.0)

        prof_before = service.get_user_profile(user_id)
        prof_rebuilt = service.rebuild_user_profile(user_id)

        assert prof_rebuilt.total_interactions == prof_before.total_interactions
        assert (
            prof_rebuilt.long_term_preferences["genre"]["animation"].score ==
            prof_before.long_term_preferences["genre"]["animation"].score
        )

