"""
src/evaluation/evaluator.py

Orchestrates the evaluation of the TechQuest movie recommender.
Strictly separates:
  1. Current System Baseline Evaluation (Runtime Semantic Relevance = UNAVAILABLE due to dimension mismatch)
  2. Offline Semantic Embedding Diagnostic (MiniLM 384d artifact check only)
  3. Candidate source tracking ('live_tmdb' vs 'local_fallback') and candidate pool recording
"""

import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from src.pipeline.recommender import MovieRecommender
from src.evaluation.metrics import (
    compute_hitrate,
    compute_ndcg,
    compute_quality_floor,
    compute_ild,
    compute_personalization,
    compute_dual_coverage,
    compute_offline_embedding_diagnostic
)


class RecommenderEvaluator:
    def __init__(
        self,
        config_path: str = "config.yaml",
        benchmark_path: str = "data/evaluation/benchmark_dataset.json"
    ):
        self.config_path = config_path
        self.benchmark_path = benchmark_path
        self.recommender = MovieRecommender(config_path=config_path)
        self.benchmark_data = self._load_benchmark()
        
        # Isolated 384d SentenceTransformer model ONLY for offline embedding diagnostic
        print("[Evaluator] Loading MiniLM 384d model for Offline Embedding Diagnostic...", flush=True)
        try:
            self.diagnostic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[Evaluator] Warning: Failed to load MiniLM diagnostic model: {e}", flush=True)
            self.diagnostic_encoder = None

    def _load_benchmark(self) -> Dict[str, Any]:
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def evaluate_baseline(self) -> Dict[str, Any]:
        """
        Executes Current System Baseline Evaluation and Offline Semantic Embedding Diagnostic.
        """
        await self.recommender.initialize()
        
        # Candidate Source & Timestamp
        is_live = self.recommender.tmdb_client is not None
        candidate_source = "live_tmdb" if is_live else "local_fallback"
        retrieval_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat() if is_live else None
        
        test_cases = self.benchmark_data.get("test_cases", [])
        results_per_case = []
        
        print(f"[Evaluator] Starting evaluation on {len(test_cases)} benchmark test cases. Source: {candidate_source}", flush=True)
        
        for tc in test_cases:
            tc_id = tc["id"]
            category = tc.get("category", "general")
            query = tc.get("query", "")
            watch_history = tc.get("watch_history", [])
            subscriptions = tc.get("subscriptions", [])
            expected_movies = tc.get("expected_relevant_movies", [])
            
            expected_map = {m["movie_id"]: m.get("relevance", 1.0) for m in expected_movies}
            
            # Execute recommender as-is
            rec_result = await self.recommender.get_recommendations(
                user_id=tc_id,
                watch_history=watch_history,
                subscriptions=subscriptions,
                top_k=10,
                region="US",
                language="en",
                natural_language=query
            )
            
            raw_recos = rec_result.get("recommendations", [])
            recommended_movies = [r["movie"] for r in raw_recos if "movie" in r]
            recommended_ids = [m["id"] for m in recommended_movies if "id" in m]
            recommended_titles = [m.get("title", "") for m in recommended_movies]
            
            # Extract candidate pool used
            # If local fallback, pool is local FeatureMatrix movies
            fm = self.recommender.feature_matrix
            if not is_live:
                candidate_pool_ids = list(fm.movie_ids) if fm else []
            else:
                # Capture candidates returned in recommendations
                candidate_pool_ids = recommended_ids
                
            # --- 1. Dual Coverage Tracking ---
            coverage_info = compute_dual_coverage(watch_history, recommended_ids, fm, k=10)
            hist_cov = coverage_info["history_input_coverage"]
            reco_cov_5 = compute_dual_coverage(watch_history, recommended_ids, fm, k=5)["recommendation_feature_coverage"]
            reco_cov_10 = coverage_info["recommendation_feature_coverage"]
            
            # --- 2. Baseline Metrics (Scope A) ---
            hitrate_5 = compute_hitrate(recommended_ids, expected_map, k=5)
            hitrate_10 = compute_hitrate(recommended_ids, expected_map, k=10)
            
            ndcg_5 = compute_ndcg(recommended_ids, expected_map, k=5)
            ndcg_10 = compute_ndcg(recommended_ids, expected_map, k=10)
            
            qf_5 = compute_quality_floor(recommended_movies, k=5)
            qf_10 = compute_quality_floor(recommended_movies, k=10)
            
            ild_5, ild_valid_5 = compute_ild(recommended_ids, fm, k=5)
            ild_10, ild_valid_10 = compute_ild(recommended_ids, fm, k=10)
            
            # Personalization / History Coherence
            pref_ext = self.recommender.preference_extractor
            user_prof = pref_ext.extract_preferences(watch_history) if (pref_ext and watch_history) else None
            pers_5, pers_num_5 = compute_personalization(recommended_ids, user_prof, pref_ext, fm, k=5)
            pers_10, pers_num_10 = compute_personalization(recommended_ids, user_prof, pref_ext, fm, k=10)
            
            # Runtime Semantic Relevance is explicitly UNAVAILABLE due to dimension mismatch
            runtime_semantic_status = "UNAVAILABLE / INVALID DUE TO EMBEDDING-SPACE MISMATCH"
            
            # --- 3. Offline Semantic Embedding Diagnostic (Scope B) ---
            minilm_diag_5, minilm_valid_5 = compute_offline_embedding_diagnostic(
                query, recommended_ids, fm, self.diagnostic_encoder, k=5
            )
            minilm_diag_10, minilm_valid_10 = compute_offline_embedding_diagnostic(
                query, recommended_ids, fm, self.diagnostic_encoder, k=10
            )
            
            case_res = {
                "test_case_id": tc_id,
                "category": category,
                "query": query,
                "candidate_source": candidate_source,
                "candidate_pool_size": len(candidate_pool_ids),
                "recommended_ids": recommended_ids,
                "recommended_titles": recommended_titles,
                "coverage": {
                    "history_input_coverage": hist_cov,
                    "recommendation_feature_coverage_at_5": reco_cov_5,
                    "recommendation_feature_coverage_at_10": reco_cov_10
                },
                "baseline_metrics": {
                    "runtime_semantic_relevance": runtime_semantic_status,
                    "hitrate_at_5": hitrate_5,
                    "hitrate_at_10": hitrate_10,
                    "ndcg_at_5": ndcg_5,
                    "ndcg_at_10": ndcg_10,
                    "avg_rating_at_5": qf_5["avg_rating"],
                    "avg_rating_at_10": qf_10["avg_rating"],
                    "avg_vote_count_at_5": qf_5["avg_vote_count"],
                    "avg_vote_count_at_10": qf_10["avg_vote_count"],
                    "rating_coverage_at_5": qf_5["rating_coverage"],
                    "rating_coverage_at_10": qf_10["rating_coverage"],
                    "ild_at_5": ild_5,
                    "ild_at_10": ild_10,
                    "personalization_at_5": pers_5 if len(watch_history) > 0 else None,
                    "personalization_at_10": pers_10 if len(watch_history) > 0 else None
                },
                "offline_embedding_diagnostic": {
                    "description": "MiniLM 384d Query-to-Item Cosine Similarity (Artifact Diagnostic Only)",
                    "minilm_cosine_sim_at_5": minilm_diag_5,
                    "minilm_cosine_sim_at_10": minilm_diag_10,
                    "evaluated_movies_count_at_5": minilm_valid_5,
                    "evaluated_movies_count_at_10": minilm_valid_10
                }
            }
            results_per_case.append(case_res)
            
        # Compute Aggregates
        aggregate_results = self._compute_aggregates(results_per_case)
        
        evaluation_payload = {
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "benchmark_version": self.benchmark_data.get("_metadata", {}).get("version", "1.0"),
                "total_cases": len(test_cases),
                "candidate_source": candidate_source,
                "retrieval_timestamp": retrieval_timestamp,
                "feature_matrix_movie_count": len(fm.movie_ids) if fm else 0,
                "configured_embedding_model": "multi-qa-mpnet-base-dot-v1",
                "configured_embedding_dim": 768,
                "stored_embedding_model": "all-MiniLM-L6-v2",
                "stored_embedding_dim": 384,
                "runtime_semantic_scoring_status": "BROKEN / DEFAULT 0.0 (ValueError on shape mismatch)",
                "semantic_weight_experiments_status": "BLOCKED (Underlying semantic score is always 0.0)"
            },
            "aggregate_results": aggregate_results,
            "per_test_case_results": results_per_case
        }
        
        return evaluation_payload

    def _compute_aggregates(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes mean aggregate metrics across test cases handling None/N/A values properly."""
        def safe_mean(vals: List[Any]) -> Optional[float]:
            valid = [float(v) for v in vals if v is not None]
            return float(np.mean(valid)) if valid else None

        hr5 = safe_mean([c["baseline_metrics"]["hitrate_at_5"] for c in case_results])
        hr10 = safe_mean([c["baseline_metrics"]["hitrate_at_10"] for c in case_results])
        
        ndcg5 = safe_mean([c["baseline_metrics"]["ndcg_at_5"] for c in case_results])
        ndcg10 = safe_mean([c["baseline_metrics"]["ndcg_at_10"] for c in case_results])
        
        r5 = safe_mean([c["baseline_metrics"]["avg_rating_at_5"] for c in case_results])
        r10 = safe_mean([c["baseline_metrics"]["avg_rating_at_10"] for c in case_results])
        
        vc5 = safe_mean([c["baseline_metrics"]["avg_vote_count_at_5"] for c in case_results])
        vc10 = safe_mean([c["baseline_metrics"]["avg_vote_count_at_10"] for c in case_results])
        
        ild5 = safe_mean([c["baseline_metrics"]["ild_at_5"] for c in case_results])
        ild10 = safe_mean([c["baseline_metrics"]["ild_at_10"] for c in case_results])
        
        pers5 = safe_mean([c["baseline_metrics"]["personalization_at_5"] for c in case_results])
        pers10 = safe_mean([c["baseline_metrics"]["personalization_at_10"] for c in case_results])
        
        hist_cov = safe_mean([c["coverage"]["history_input_coverage"] for c in case_results])
        reco_cov5 = safe_mean([c["coverage"]["recommendation_feature_coverage_at_5"] for c in case_results])
        reco_cov10 = safe_mean([c["coverage"]["recommendation_feature_coverage_at_10"] for c in case_results])
        
        diag_minilm5 = safe_mean([c["offline_embedding_diagnostic"]["minilm_cosine_sim_at_5"] for c in case_results])
        diag_minilm10 = safe_mean([c["offline_embedding_diagnostic"]["minilm_cosine_sim_at_10"] for c in case_results])
        
        return {
            "runtime_semantic_relevance": "UNAVAILABLE / INVALID DUE TO EMBEDDING-SPACE MISMATCH",
            "hitrate_at_5": hr5,
            "hitrate_at_10": hr10,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10,
            "avg_rating_at_5": r5,
            "avg_rating_at_10": r10,
            "avg_vote_count_at_5": vc5,
            "avg_vote_count_at_10": vc10,
            "ild_at_5": ild5,
            "ild_at_10": ild10,
            "personalization_at_5": pers5,
            "personalization_at_10": pers10,
            "history_input_coverage": hist_cov,
            "recommendation_feature_coverage_at_5": reco_cov5,
            "recommendation_feature_coverage_at_10": reco_cov10,
            "offline_minilm_diagnostic_at_5": diag_minilm5,
            "offline_minilm_diagnostic_at_10": diag_minilm10
        }
