"""
tests/test_evaluation.py

Unit tests for evaluation framework metrics, coverage calculations, and benchmark dataset schema.
"""

import json
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_cosine_similarity,
    compute_hitrate,
    compute_ndcg,
    compute_quality_floor,
    compute_ild,
    compute_dual_coverage
)


def test_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    v_zero = np.array([0.0, 0.0, 0.0])
    
    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0)
    assert compute_cosine_similarity(v1, v_zero) == 0.0


def test_hitrate_and_ndcg():
    recs = [101, 102, 103, 104, 105]
    rel_map = {101: 3.0, 103: 2.0}
    
    # HitRate
    assert compute_hitrate(recs, rel_map, k=5) == 1.0
    assert compute_hitrate([999], rel_map, k=5) == 0.0
    
    # NDCG
    ndcg5 = compute_ndcg(recs, rel_map, k=5)
    assert ndcg5 is not None
    assert 0.0 <= ndcg5 <= 1.0
    
    # Perfect ranking NDCG
    perfect_recs = [101, 103]
    perfect_ndcg = compute_ndcg(perfect_recs, rel_map, k=2)
    assert perfect_ndcg == pytest.approx(1.0)
    
    # Missing ground truth -> None
    assert compute_ndcg(recs, {}, k=5) is None


def test_quality_floor():
    movies = [
        {"id": 1, "vote_average": 8.5, "vote_count": 1000},
        {"id": 2, "vote_average": 7.0, "vote_count": 500},
        {"id": 3, "vote_average": None, "vote_count": 0}
    ]
    
    qf = compute_quality_floor(movies, k=3)
    assert qf["avg_rating"] == pytest.approx(7.75)
    assert qf["avg_vote_count"] == pytest.approx(500.0)
    assert qf["rating_coverage"] == pytest.approx(2/3)


class DummyFeatureMatrix:
    def __init__(self):
        self.movie_ids = [101, 102, 103]
        self.semantic_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])


def test_ild_diversity():
    fm = DummyFeatureMatrix()
    
    # Pairwise distance between [1.0, 0.0] and [0.0, 1.0] is 1.0 - 0.0 = 1.0
    ild, n = compute_ild([101, 102], fm, k=2)
    assert n == 2
    assert ild == pytest.approx(1.0)
    
    # Single item
    ild_single, n_single = compute_ild([101], fm, k=1)
    assert n_single == 1
    assert ild_single == 0.0
    
    # Items absent from FM
    ild_absent, n_absent = compute_ild([999, 888], fm, k=2)
    assert n_absent == 0
    assert ild_absent is None


def test_dual_coverage():
    fm = DummyFeatureMatrix()
    watch_history = [{"movie_id": 101}, {"movie_id": 999}]
    recommended_ids = [101, 102, 888, 777]
    
    cov = compute_dual_coverage(watch_history, recommended_ids, fm, k=4)
    # 1 of 2 watch history items in FM -> 0.5
    assert cov["history_input_coverage"] == pytest.approx(0.5)
    # 2 of 4 recommended items in FM -> 0.5
    assert cov["recommendation_feature_coverage"] == pytest.approx(0.5)


def test_benchmark_dataset_validity():
    with open("data/evaluation/benchmark_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "_metadata" in data
    assert "test_cases" in data
    cases = data["test_cases"]
    assert len(cases) == 25
    
    categories = set(c["category"] for c in cases)
    assert len(categories) == 5
    
    for c in cases:
        assert "id" in c
        assert "query" in c
        assert "expected_relevant_movies" in c
        for m in c["expected_relevant_movies"]:
            assert isinstance(m["movie_id"], int)
            assert "relevance" in m
