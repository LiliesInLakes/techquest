"""
evaluate_recommender.py

CLI entry point for running TechQuest Movie Recommender Evaluation.

Usage:
    python evaluate_recommender.py [--config config.yaml] [--benchmark data/evaluation/benchmark_dataset.json]
"""

import argparse
import asyncio
import sys
from pathlib import Path

from src.evaluation.evaluator import RecommenderEvaluator
from src.evaluation.report_generator import ReportGenerator


async def main_async(args):
    print("=" * 60)
    print(" TechQuest Movie Recommender — Evaluation Framework")
    print("=" * 60)
    
    evaluator = RecommenderEvaluator(
        config_path=args.config,
        benchmark_path=args.benchmark
    )
    
    print("\n[Evaluation] Running Current System Baseline & Offline Diagnostic...")
    payload = await evaluator.evaluate_baseline()
    
    print("\n[Report] Generating evaluation.md, JSON, and CSV artifacts...")
    generator = ReportGenerator(
        output_dir=args.output_dir,
        report_path=args.report_path
    )
    md_path, json_path, csv_path = generator.generate_all(payload)
    
    meta = payload["metadata"]
    agg = payload["aggregate_results"]
    
    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETED")
    print("=" * 60)
    print(f" Candidate Source            : {meta.get('candidate_source')}")
    print(f" Runtime Semantic Relevance  : {agg.get('runtime_semantic_relevance')}")
    print(f" HitRate@5                   : {agg.get('hitrate_at_5')}")
    print(f" HitRate@10                  : {agg.get('hitrate_at_10')}")
    print(f" NDCG@5                      : {agg.get('ndcg_at_5')}")
    print(f" NDCG@10                     : {agg.get('ndcg_at_10')}")
    print(f" Avg Vote Average@5          : {agg.get('avg_rating_at_5')}")
    print(f" ILD@5 (Diversity)           : {agg.get('ild_at_5')}")
    print(f" Personalization@5           : {agg.get('personalization_at_5')}")
    print(f" Offline MiniLM Diag @5      : {agg.get('offline_minilm_diagnostic_at_5')}")
    print("-" * 60)
    print(f" Primary Report Written To   : {md_path.resolve()}")
    print(f" Machine-Readable JSON       : {json_path.resolve()}")
    print(f" Machine-Readable CSV        : {csv_path.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TechQuest Movie Recommender")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--benchmark", default="data/evaluation/benchmark_dataset.json", help="Path to benchmark_dataset.json")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory for JSON/CSV outputs")
    parser.add_argument("--report-path", default="evaluation.md", help="Path for generated evaluation.md")
    
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
