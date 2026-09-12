import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_mcp_server():
    import uvicorn
    from mcp_server import app
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_a2a_agent():
    from a2a_protocol import MovieRecommenderA2AAgent
    agent = MovieRecommenderA2AAgent()
    print(f"A2A Agent started: {agent.agent_info.name} ({agent.agent_info.agent_id})")
    print("Capabilities:", [c.name for c in agent.agent_info.capabilities])


async def run_recommendation_cli(
    user_id: str,
    genre: str = None,
    region: str = "US",
    language: str = "en",
    subscriptions: list = None,
    watch_history: list = None,
    top_k: int = 10,
    natural_language: str = None
):
    from src.pipeline.recommender import MovieRecommender

    recommender = MovieRecommender()

    if watch_history:
        result = await recommender.get_recommendations(
            user_id=user_id,
            watch_history=watch_history,
            subscriptions=subscriptions or [],
            region=region,
            language=language,
            top_k=top_k,
            natural_language=natural_language
        )
        print(json.dumps(result, indent=2))
    else:
        movies = await recommender.search_movies(
            genre=genre,
            region=region,
            language=language,
            subscriptions=subscriptions or [],
            max_results=top_k
        )
        print(json.dumps({"movies": movies}, indent=2))


def run_precompute():
    from src.features.matrix_builder import FeatureMatrixBuilder

    builder = FeatureMatrixBuilder()
    builder.build_and_save()
    print("Feature matrix built and saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Movie Recommender Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("mcp", help="Run MCP server")
    subparsers.add_parser("a2a", help="Run A2A agent")
    subparsers.add_parser("precompute", help="Precompute feature matrix")

    cli_parser = subparsers.add_parser("recommend", help="Get recommendations via CLI")
    cli_parser.add_argument("--user-id", required=True, help="User ID")
    cli_parser.add_argument("--genre", help="Genre filter")
    cli_parser.add_argument("--region", default="US", help="Region code")
    cli_parser.add_argument("--language", default="en", help="Language code")
    cli_parser.add_argument("--subscriptions", nargs="+", default=[], help="Subscription services")
    cli_parser.add_argument("--watch-history", type=str, help="Path to watch history JSON")
    cli_parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    cli_parser.add_argument("--natural-language", type=str, help="Natural language query (e.g., 'funny 90s space comedy')")

    args = parser.parse_args()

    if args.command == "mcp":
        run_mcp_server()
    elif args.command == "a2a":
        run_a2a_agent()
    elif args.command == "precompute":
        run_precompute()
    elif args.command == "recommend":
        watch_history = None
        if args.watch_history:
            with open(args.watch_history) as f:
                watch_history = json.load(f)

        asyncio.run(run_recommendation_cli(
            user_id=args.user_id,
            genre=args.genre,
            region=args.region,
            language=args.language,
            subscriptions=args.subscriptions,
            watch_history=watch_history,
            top_k=args.top_k,
            natural_language=args.natural_language
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
