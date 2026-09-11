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


async def run_business_recommendation_cli(
    user_id: str,
    partner_id: str,
    genre: str = None,
    region: str = "US",
    language: str = "en",
    subscriptions: list = None,
    watch_history: list = None,
    top_k: int = 10,
    natural_language: str = None
):
    from src.pipeline.recommender import BusinessMovieRecommender
    
    recommender = BusinessMovieRecommender()
    
    if watch_history:
        result = await recommender.get_recommendations_with_business_context(
            user_id=user_id,
            partner_id=partner_id,
            watch_history=watch_history,
            subscriptions=subscriptions or [],
            region=region,
            language=language,
            top_k=top_k,
            natural_language=natural_language
        )
        print(json.dumps(result, indent=2))
        recommender.shutdown()
    else:
        movies = await recommender.search_movies(
            genre=genre,
            region=region,
            language=language,
            subscriptions=subscriptions or [],
            max_results=top_k
        )
        print(json.dumps({"movies": movies}, indent=2))
        recommender.shutdown()


async def run_churn_assessment_cli(
    user_id: str,
    partner_id: str,
    user_data_path: str = None
):
    from src.pipeline.recommender import BusinessMovieRecommender
    
    recommender = BusinessMovieRecommender()
    
    user_data = {}
    if user_data_path:
        with open(user_data_path) as f:
            user_data = json.load(f)
    
    result = await recommender.assess_churn_risk(user_id, partner_id, user_data)
    print(json.dumps(result, indent=2))
    recommender.shutdown()


async def run_trigger_action_cli(
    user_id: str,
    partner_id: str,
    action_type: str,
    trigger: str,
    payload: dict,
    reasoning: str,
    priority: str = "medium"
):
    from src.pipeline.recommender import BusinessMovieRecommender
    
    recommender = BusinessMovieRecommender()
    
    result = await recommender.trigger_business_action(
        action_type=action_type,
        partner_id=partner_id,
        user_id=user_id,
        trigger=trigger,
        payload=payload,
        reasoning=reasoning,
        priority=priority
    )
    print(json.dumps(result, indent=2))
    recommender.shutdown()


async def run_ingest_events_cli(
    events_path: str
):
    from src.pipeline.recommender import BusinessMovieRecommender
    
    recommender = BusinessMovieRecommender()
    
    with open(events_path) as f:
        events = json.load(f)
    
    result = await recommender.ingest_events(events)
    print(json.dumps(result, indent=2))
    recommender.shutdown()


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
    
    cli_parser = subparsers.add_parser("recommend", help="Get recommendations via CLI (basic)")
    cli_parser.add_argument("--user-id", required=True, help="User ID")
    cli_parser.add_argument("--genre", help="Genre filter")
    cli_parser.add_argument("--region", default="US", help="Region code")
    cli_parser.add_argument("--language", default="en", help="Language code")
    cli_parser.add_argument("--subscriptions", nargs="+", default=[], help="Subscription services")
    cli_parser.add_argument("--watch-history", type=str, help="Path to watch history JSON")
    cli_parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    cli_parser.add_argument("--natural-language", type=str, help="Natural language query (e.g., 'funny 90s space comedy')")
    
    biz_parser = subparsers.add_parser("recommend-biz", help="Get recommendations with business context")
    biz_parser.add_argument("--user-id", required=True, help="User ID")
    biz_parser.add_argument("--partner-id", required=True, help="Partner ID (e.g., airline_default, hotel_default)")
    biz_parser.add_argument("--genre", help="Genre filter")
    biz_parser.add_argument("--region", default="US", help="Region code")
    biz_parser.add_argument("--language", default="en", help="Language code")
    biz_parser.add_argument("--subscriptions", nargs="+", default=[], help="Subscription services")
    biz_parser.add_argument("--watch-history", type=str, help="Path to watch history JSON")
    biz_parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    biz_parser.add_argument("--natural-language", type=str, help="Natural language query (e.g., 'funny 90s space comedy')")
    
    churn_parser = subparsers.add_parser("assess-churn", help="Assess user churn risk")
    churn_parser.add_argument("--user-id", required=True, help="User ID")
    churn_parser.add_argument("--partner-id", required=True, help="Partner ID")
    churn_parser.add_argument("--user-data", type=str, help="Path to user data JSON")
    
    action_parser = subparsers.add_parser("trigger-action", help="Trigger a business action")
    action_parser.add_argument("--user-id", required=True, help="User ID")
    action_parser.add_argument("--partner-id", required=True, help="Partner ID")
    action_parser.add_argument("--action-type", required=True, help="Action type (e.g., recommend_movie, offer_bundle_upgrade)")
    action_parser.add_argument("--trigger", required=True, help="Trigger type (e.g., session_start, churn_risk_high)")
    action_parser.add_argument("--payload", type=str, help="JSON payload for action")
    action_parser.add_argument("--reasoning", required=True, help="Reasoning for action")
    action_parser.add_argument("--priority", default="medium", help="Priority (low, medium, high, critical)")
    
    events_parser = subparsers.add_parser("ingest-events", help="Ingest analytics events")
    events_parser.add_argument("--events-file", required=True, help="Path to events JSON file")
    
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
    elif args.command == "recommend-biz":
        watch_history = None
        if args.watch_history:
            with open(args.watch_history) as f:
                watch_history = json.load(f)
        
        asyncio.run(run_business_recommendation_cli(
            user_id=args.user_id,
            partner_id=args.partner_id,
            genre=args.genre,
            region=args.region,
            language=args.language,
            subscriptions=args.subscriptions,
            watch_history=watch_history,
            top_k=args.top_k,
            natural_language=args.natural_language
        ))
    elif args.command == "assess-churn":
        asyncio.run(run_churn_assessment_cli(
            user_id=args.user_id,
            partner_id=args.partner_id,
            user_data_path=args.user_data
        ))
    elif args.command == "trigger-action":
        payload = json.loads(args.payload) if args.payload else {}
        asyncio.run(run_trigger_action_cli(
            user_id=args.user_id,
            partner_id=args.partner_id,
            action_type=args.action_type,
            trigger=args.trigger,
            payload=payload,
            reasoning=args.reasoning,
            priority=args.priority
        ))
    elif args.command == "ingest-events":
        asyncio.run(run_ingest_events_cli(
            events_path=args.events_file
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
