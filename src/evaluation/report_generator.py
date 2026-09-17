"""
src/evaluation/report_generator.py

Automated report generator for evaluation results:
  - Formats and writes primary evaluation report to evaluation.md
  - Writes machine-readable JSON and CSV files to evaluation_results/
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


class ReportGenerator:
    def __init__(self, output_dir: str = "evaluation_results", report_path: str = "evaluation.md"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = Path(report_path)

    def generate_all(self, evaluation_payload: Dict[str, Any]) -> Tuple[Path, Path, Path]:
        """Generates evaluation.md, JSON, and CSV artifacts."""
        md_path = self.generate_markdown_report(evaluation_payload)
        json_path = self.save_json_results(evaluation_payload)
        csv_path = self.save_csv_results(evaluation_payload)
        return md_path, json_path, csv_path

    def save_json_results(self, payload: Dict[str, Any]) -> Path:
        json_file = self.output_dir / "baseline_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return json_file

    def save_csv_results(self, payload: Dict[str, Any]) -> Path:
        csv_file = self.output_dir / "baseline_results.csv"
        cases = payload.get("per_test_case_results", [])
        
        headers = [
            "test_case_id", "category", "query", "candidate_source", "candidate_pool_size",
            "recommended_ids", "recommended_titles",
            "history_input_coverage", "recommendation_feature_coverage_at_5", "recommendation_feature_coverage_at_10",
            "runtime_semantic_relevance", "hitrate_at_5", "hitrate_at_10", "ndcg_at_5", "ndcg_at_10",
            "avg_rating_at_5", "avg_rating_at_10", "avg_vote_count_at_5", "avg_vote_count_at_10",
            "ild_at_5", "ild_at_10", "personalization_at_5", "personalization_at_10",
            "minilm_diagnostic_sim_at_5", "minilm_diagnostic_sim_at_10"
        ]
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for c in cases:
                cov = c.get("coverage", {})
                bm = c.get("baseline_metrics", {})
                diag = c.get("offline_embedding_diagnostic", {})
                writer.writerow([
                    c.get("test_case_id"),
                    c.get("category"),
                    c.get("query"),
                    c.get("candidate_source"),
                    c.get("candidate_pool_size"),
                    ";".join(map(str, c.get("recommended_ids", []))),
                    ";".join(c.get("recommended_titles", [])),
                    cov.get("history_input_coverage"),
                    cov.get("recommendation_feature_coverage_at_5"),
                    cov.get("recommendation_feature_coverage_at_10"),
                    bm.get("runtime_semantic_relevance"),
                    bm.get("hitrate_at_5"),
                    bm.get("hitrate_at_10"),
                    bm.get("ndcg_at_5"),
                    bm.get("ndcg_at_10"),
                    bm.get("avg_rating_at_5"),
                    bm.get("avg_rating_at_10"),
                    bm.get("avg_vote_count_at_5"),
                    bm.get("avg_vote_count_at_10"),
                    bm.get("ild_at_5"),
                    bm.get("ild_at_10"),
                    bm.get("personalization_at_5"),
                    bm.get("personalization_at_10"),
                    diag.get("minilm_cosine_sim_at_5"),
                    diag.get("minilm_cosine_sim_at_10")
                ])
        return csv_file

    def generate_markdown_report(self, payload: Dict[str, Any]) -> Path:
        meta = payload.get("metadata", {})
        agg = payload.get("aggregate_results", {})
        cases = payload.get("per_test_case_results", [])
        
        def fmt(val: Any, decimals: int = 4) -> str:
            if val is None:
                return "N/A"
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
            return str(val)

        cand_source_label = meta.get("candidate_source", "unknown")
        is_fallback = cand_source_label == "local_fallback"
        source_display = "Live TMDB Discovery API" if not is_fallback else "Local Fallback (FeatureMatrix Pool)"
        
        lines = []
        lines.append("# TechQuest Movie Recommender — Evaluation Report")
        lines.append("")
        lines.append(f"**Evaluation Timestamp (UTC)**: `{meta.get('timestamp')}`")
        lines.append(f"**Benchmark Dataset Version**: `{meta.get('benchmark_version')}` ({meta.get('total_cases')} test cases)")
        lines.append(f"**Candidate Source**: `{source_display}` (`{cand_source_label}`)")
        if meta.get("retrieval_timestamp"):
            lines.append(f"**API Retrieval Timestamp**: `{meta.get('retrieval_timestamp')}`")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # PROMINENT SECTION: Data & Embedding Integrity
        lines.append("## 1. Data & Embedding Integrity")
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append("> **Embedding Space Mismatch & Runtime Scoring Status**")
        lines.append("> - **Configured Query Embedding Model**: `multi-qa-mpnet-base-dot-v1` (**768-dimensional**)")
        lines.append("> - **Stored Movie Embedding Model**: `all-MiniLM-L6-v2` (**384-dimensional**)")
        lines.append("> - **Runtime Behavior**: `MovieRanker._compute_semantic_score()` catches `ValueError` due to shape mismatch (`(1, 768)` vs `(1, 384)`) and silently defaults `semantic_score` to `0.0` for all movies.")
        lines.append("> - **Runtime Semantic Relevance Status**: **BROKEN / DEFAULT 0.0**. Reported as `UNAVAILABLE / INVALID` below.")
        lines.append("")
        lines.append("### On-Disk Artifact Audit Summary")
        lines.append("| Audit Item | Value / Description |")
        lines.append("|---|---|")
        lines.append(f"| **FeatureMatrix Movie Count** | `{meta.get('feature_matrix_movie_count')}` movies |")
        lines.append(f"| **Configured Query Model** | `multi-qa-mpnet-base-dot-v1` (768d) |")
        lines.append(f"| **Stored Movie Model** | `all-MiniLM-L6-v2` (384d) |")
        lines.append(f"| **Runtime Semantic Score** | Always `0.0` (Caught `ValueError`) |")
        lines.append(f"| **Candidate Discovery Mode** | `{source_display}` |")
        lines.append(f"| **History Input Coverage** | `{fmt(agg.get('history_input_coverage') * 100 if agg.get('history_input_coverage') is not None else None, 2)}%` |")
        lines.append(f"| **Reco Feature Coverage @5** | `{fmt(agg.get('recommendation_feature_coverage_at_5') * 100 if agg.get('recommendation_feature_coverage_at_5') is not None else None, 2)}%` |")
        lines.append(f"| **Reco Feature Coverage @10** | `{fmt(agg.get('recommendation_feature_coverage_at_10') * 100 if agg.get('recommendation_feature_coverage_at_10') is not None else None, 2)}%` |")
        lines.append("")
        
        if is_fallback:
            lines.append("> [!IMPORTANT]")
            lines.append("> **Local Fallback Evaluation Note**")
            lines.append("> This evaluation ran using the **Local Fallback Candidate Pool** because live TMDB API client was uninitialized or key missing. Candidate discovery was restricted to the 10 movies in `feature_matrix.pkl`.")
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
        # SECTION 2: Current System Baseline Results
        lines.append("## 2. Current System Baseline Evaluation")
        lines.append("")
        lines.append("This table presents the objective measurements of the current recommender **operating exactly as-is**.")
        lines.append("")
        lines.append("| Metric Dimension | Metric | @5 Value | @10 Value | Status / Note |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| **Semantic Relevance** | Runtime Query-Item Cosine Sim | `UNAVAILABLE` | `UNAVAILABLE` | Invalid due to 768d vs 384d mismatch |")
        lines.append(f"| **Ranking Quality** | HitRate@K | `{fmt(agg.get('hitrate_at_5'))}` | `{fmt(agg.get('hitrate_at_10'))}` | Ground truth hit rate |")
        lines.append(f"| **Ranking Quality** | NDCG@K | `{fmt(agg.get('ndcg_at_5'))}` | `{fmt(agg.get('ndcg_at_10'))}` | Graded relevance DCG/IDCG |")
        lines.append(f"| **Quality Floor** | Avg Vote Average@K | `{fmt(agg.get('avg_rating_at_5'))}` | `{fmt(agg.get('avg_rating_at_10'))}` | Mean IMDb vote average |")
        lines.append(f"| **Quality Floor** | Avg Vote Count@K | `{fmt(agg.get('avg_vote_count_at_5'), 1)}` | `{fmt(agg.get('avg_vote_count_at_10'), 1)}` | Mean vote volume |")
        lines.append(f"| **Diversity** | Intra-List Distance ILD@K | `{fmt(agg.get('ild_at_5'))}` | `{fmt(agg.get('ild_at_10'))}` | Pairwise cosine distance |")
        lines.append(f"| **History Coherence** | Personalization Score@K | `{fmt(agg.get('personalization_at_5'))}` | `{fmt(agg.get('personalization_at_10'))}` | History content similarity |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # SECTION 3: Offline Semantic Embedding Diagnostic
        lines.append("## 3. Offline Semantic Embedding Diagnostic")
        lines.append("")
        lines.append("> [!NOTE]")
        lines.append("> **Diagnostic Scope & Methodology**")
        lines.append("> This is an **isolated artifact diagnostic** measuring query-to-item cosine similarity using `all-MiniLM-L6-v2` (384d) against stored movie embeddings in `feature_matrix.pkl`.")
        lines.append("> **This diagnostic does NOT represent runtime recommender performance**, because runtime `MovieRanker` uses 768-dim query embeddings and defaults semantic scores to 0.0.")
        lines.append("")
        lines.append("| Diagnostic Metric | @5 Value | @10 Value | Scope |")
        lines.append("|---|---|---|---|")
        lines.append(f"| **MiniLM (384d) Cosine Similarity** | `{fmt(agg.get('offline_minilm_diagnostic_at_5'))}` | `{fmt(agg.get('offline_minilm_diagnostic_at_10'))}` | Stored `feature_matrix.pkl` artifact check |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # SECTION 4: Per-Test-Case Breakdown
        lines.append("## 4. Per-Test-Case Detailed Breakdown")
        lines.append("")
        lines.append("| ID | Category | Query | Source | Top-5 Recommended Titles | Reco Cov @5 | HitRate@5 | NDCG@5 | MiniLM Diag @5 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in cases:
            bm = c.get("baseline_metrics", {})
            cov = c.get("coverage", {})
            diag = c.get("offline_embedding_diagnostic", {})
            titles = ", ".join(c.get("recommended_titles", [])[:5])
            lines.append(
                f"| `{c.get('test_case_id')}` | `{c.get('category')}` | {c.get('query')[:40]}... | `{c.get('candidate_source')}` | {titles[:45]} | `{fmt(cov.get('recommendation_feature_coverage_at_5') * 100, 0)}%` | `{fmt(bm.get('hitrate_at_5'), 2)}` | `{fmt(bm.get('ndcg_at_5'), 4)}` | `{fmt(diag.get('minilm_cosine_sim_at_5'), 4)}` |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # SECTION 5: Failure Cases & Root Cause Analysis
        lines.append("## 5. Failure Cases & Root Cause Analysis")
        lines.append("")
        weak_cases = [c for c in cases if c.get("baseline_metrics", {}).get("ndcg_at_5") is not None and c["baseline_metrics"]["ndcg_at_5"] < 0.5]
        if not weak_cases:
            lines.append("No test cases returned NDCG@5 < 0.5.")
        else:
            for wc in weak_cases:
                lines.append(f"### Test Case `{wc['test_case_id']}`: \"{wc['query']}\"")
                lines.append(f"- **Category**: `{wc['category']}`")
                lines.append(f"- **NDCG@5**: `{fmt(wc['baseline_metrics']['ndcg_at_5'])}` | **HitRate@5**: `{fmt(wc['baseline_metrics']['hitrate_at_5'])}`")
                lines.append(f"- **Recommended Titles**: {', '.join(wc['recommended_titles'][:5])}")
                lines.append(f"- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.")
                lines.append("")
                
        lines.append("---")
        lines.append("")
        
        # SECTION 6: System & Dataset Limitations
        lines.append("## 6. System & Dataset Limitations")
        lines.append("")
        lines.append("1. **10-Movie FeatureMatrix Pool**: `feature_matrix.pkl` contains only 10 movies. When TMDB API is offline or uninitialized, local fallback recommendations are severely constrained.")
        lines.append("2. **Embedding Model Incompatibility**: `config.yaml` specifies 768-dim `multi-qa-mpnet-base-dot-v1`, but stored `feature_matrix.pkl` contains 384-dim `all-MiniLM-L6-v2` embeddings, rendering semantic scoring completely inactive at runtime.")
        lines.append("3. **Candidate Coverage Gap**: Live TMDB recommendations not present in `feature_matrix.pkl` cannot be evaluated for embedding-dependent ILD or Personalization metrics.")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # SECTION 7: Future Semantic-Weight Experiment Protocol
        lines.append("## 7. Future Semantic-Weight Experiment Protocol")
        lines.append("")
        lines.append("> [!IMPORTANT]")
        lines.append("> **Blocked Experiment Status**")
        lines.append("> `semantic_weight` experiments (`0.1`, `0.4`, `0.7`, `0.9`) are **BLOCKED** on the current system baseline because `semantic_score` is always `0.0`. Tuning weights on a zero score yields zero output change.")
        lines.append("")
        lines.append("### Protocol for Post-Repair Execution")
        lines.append("Once the embedding pipeline is repaired (so query embeddings and movie embeddings use identical models and vector dimensions), run the following protocol:")
        lines.append("1. **Configurations to Evaluate**: `semantic_weight = [0.1, 0.4, 0.7, 0.9]`")
        lines.append("2. **Fixed Environment**: Identical 25 benchmark test cases, identical ground truth, identical candidate policies.")
        lines.append("3. **Target Measurement**: Quantify trade-offs between Semantic Relevance, Ranking Quality (NDCG), and Diversity (ILD).")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # SECTION 8: Executive Summary & Final Assessment
        lines.append("## 8. Executive Summary & Final Assessment")
        lines.append("")
        lines.append("- **System Status**: The recommender's semantic search path is currently broken due to a 768d vs 384d embedding dimension mismatch.")
        lines.append("- **Ranking & Quality Baseline**: Recommender performance currently relies primarily on IMDb ratings, popularity, and TF-IDF keyword match.")
        lines.append("- **Evaluation Integrity**: Runtime Semantic Relevance is accurately marked `UNAVAILABLE`, avoiding manufactured numbers. Offline artifact quality is documented via isolated MiniLM diagnostic.")
        lines.append("")
        
        content = "\n".join(lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return self.report_path
