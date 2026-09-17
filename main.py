import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_ui(port: int = 8080, host: str = "127.0.0.1"):
    import uvicorn
    from src.ui.web_app import app
    print(f"\n=======================================================")
    print(f"  Movie Recommender UI started at http://{host}:{port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port)


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
    top_k: int = 1,
    natural_language: str = None
):
    from src.pipeline.recommender import MovieRecommender

    recommender = MovieRecommender()

    result = await recommender.get_recommendations(
        user_id=user_id,
        watch_history=watch_history or [],
        subscriptions=subscriptions or [],
        region=region,
        language=language,
        top_k=top_k,
        natural_language=natural_language
    )

    if top_k == 1 and result.get("recommendation"):
        rec = result["recommendation"]
        m = rec.get("movie", {})
        year = str(m.get("release_date", ""))[:4]
        genres = [g if isinstance(g, str) else g.get('name') for g in m.get('genres', [])]
        print("\n" + "=" * 65)
        print(f" [FINAL RECOMMENDATION]: {m.get('title')} ({year})")
        print(f" [Score]: {rec.get('score')} | [Rating]: {m.get('vote_average')}/10 ({m.get('vote_count', 0)} votes)")
        if genres:
            print(f" [Genres]: {', '.join(filter(None, genres))}")
        if m.get("streaming_platforms"):
            print(f" [Streaming]: {', '.join(m.get('streaming_platforms'))}")
        print("-" * 65)
        if rec.get("verdict"):
            print(f" [AI Verdict]:\n {rec.get('verdict')}\n")
        if rec.get("reasoning"):
            print(f" [Why You'll Like It]:\n {rec.get('reasoning')}")
        print("=" * 65)
        mid = m.get('id') or m.get('movie_id')
        print(f" Don't want to watch this movie? Get a new suggestion:")
        print(f"   python main.py reject --user-id {user_id} --movie-id {mid}")
        print("=" * 65 + "\n")
    else:
        print(json.dumps(result, indent=2))


async def run_reject_cli(
    user_id: str,
    movie_id: int,
    reason: str = "User skipped/rejected",
    natural_language: str = None
):
    from src.memory.service import UserMemoryService
    from src.pipeline.recommender import MovieRecommender

    mem = UserMemoryService()
    event, prof = mem.record_rejection(user_id=user_id, movie_id=movie_id, reason=reason)
    print(f"\n[Recorded Rejection for Movie #{movie_id}]")
    print(f"Taste penalties updated in user memory profile.")

    recommender = MovieRecommender()
    replacement = await recommender.get_replacement_recommendation(
        user_id=user_id,
        excluded_movie_ids=[movie_id],
        natural_language=natural_language
    )

    if replacement and replacement.get("movie"):
        m = replacement["movie"]
        year = str(m.get("release_date", ""))[:4]
        genres = [g if isinstance(g, str) else g.get('name') for g in m.get('genres', [])]
        print("\n" + "=" * 65)
        print(f" [NEW RECOMMENDATION]: {m.get('title')} ({year})")
        print(f" [Score]: {replacement.get('score')} | [Rating]: {m.get('vote_average')}/10")
        if genres:
            print(f" [Genres]: {', '.join(filter(None, genres))}")
        print("-" * 65)
        if replacement.get("verdict"):
            print(f" [AI Verdict]:\n {replacement.get('verdict')}\n")
        if replacement.get("reasoning"):
            print(f" [Why You'll Like It]:\n {replacement.get('reasoning')}")
        print("=" * 65 + "\n")
    else:
        print("No replacement recommendation available for the current query.\n")


def run_precompute(batch_size: int = 2000, encode_batch_size: int = None):
    from src.features.matrix_builder import FeatureMatrixBuilder

    builder = FeatureMatrixBuilder()
    builder.build_and_save(batch_size=batch_size, encode_batch_size=encode_batch_size)
    print("Feature matrix built and saved successfully")


def run_setup_db(db_path: str = "movie_features.db"):
    from src.db.connection import MovieDatabase
    db = MovieDatabase(db_path=db_path)
    valid, msg = db.verify_integrity()
    if valid:
        print(f"\n[Database Setup] SUCCESS: {msg}")
    else:
        print(f"\n[Database Setup] WARNING: {msg}")


def run_profile_inspect(user_id: str):
    from src.memory.service import UserMemoryService
    service = UserMemoryService()
    profile = service.get_user_profile(user_id)
    print(service._serialize_profile(profile))


def run_log_event(user_id: str, movie_id: int, event_type: str):
    from src.memory.service import UserMemoryService
    service = UserMemoryService()
    ev = service.record_interaction(user_id=user_id, movie_id=movie_id, event_type=event_type)
    print(json.dumps({
        "status": "success",
        "event_id": ev.id,
        "event_type": ev.event_type.value,
        "signal_weight": ev.signal_weight
    }, indent=2))


def run_rate(user_id: str, movie_id: int, rating: float, review_text: str = None):
    from src.memory.service import UserMemoryService
    service = UserMemoryService()
    r = service.record_rating(user_id=user_id, movie_id=movie_id, rating=rating, review_text=review_text)
    print(json.dumps({
        "status": "success",
        "rating_id": r.id,
        "rating": r.rating,
        "signal_target": r.signal_target
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Movie Recommender Agent with Memory System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    ui_parser = subparsers.add_parser("ui", help="Launch interactive web UI")
    ui_parser.add_argument("--port", type=int, default=8080, help="Port to run web UI on")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host address")

    subparsers.add_parser("mcp", help="Run MCP server")
    subparsers.add_parser("a2a", help="Run A2A agent")
    precomp_parser = subparsers.add_parser("precompute", help="Precompute feature matrix from movie_features.db")
    precomp_parser.add_argument("--batch-size", type=int, default=2000, help="Number of movies per checkpoint chunk (default: 2000)")
    precomp_parser.add_argument("--encode-batch-size", type=int, default=None, help="Model inference mini-batch size (e.g. 128, 256, 512)")

    setup_parser = subparsers.add_parser("setup-db", help="Verify or setup movie_features.db")
    setup_parser.add_argument("--db-path", default="movie_features.db", help="Path to SQLite database file")

    cli_parser = subparsers.add_parser("recommend", help="Get recommendations via CLI")
    cli_parser.add_argument("--user-id", required=True, help="User ID")
    cli_parser.add_argument("--genre", help="Genre filter")
    cli_parser.add_argument("--region", default="US", help="Region code")
    cli_parser.add_argument("--language", default="en", help="Language code")
    cli_parser.add_argument("--subscriptions", nargs="+", default=[], help="Subscription services")
    cli_parser.add_argument("--watch-history", type=str, help="Path to watch history JSON")
    cli_parser.add_argument("--top-k", type=int, default=1, help="Number of recommendations (default: 1)")
    cli_parser.add_argument("--natural-language", type=str, help="Natural language query (e.g., 'funny 90s space comedy')")

    reject_parser = subparsers.add_parser("reject", help="Reject a movie, learn negative preferences, and suggest a replacement")
    reject_parser.add_argument("--user-id", required=True, help="User ID")
    reject_parser.add_argument("--movie-id", type=int, required=True, help="TMDB Movie ID")
    reject_parser.add_argument("--reason", default="User skipped/rejected", help="Optional rejection reason")
    reject_parser.add_argument("--natural-language", type=str, help="Natural language query")

    prof_parser = subparsers.add_parser("profile", help="View user profile memory")
    prof_parser.add_argument("--user-id", required=True, help="User ID")
    prof_parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    event_parser = subparsers.add_parser("log-event", help="Log an observable watch event")
    event_parser.add_argument("--user-id", required=True, help="User ID")
    event_parser.add_argument("--movie-id", type=int, required=True, help="TMDB Movie ID")
    event_parser.add_argument("--event", required=True, choices=["movie_clicked", "movie_viewed", "added_to_watchlist", "marked_watched"], help="Event type")

    rate_parser = subparsers.add_parser("rate", help="Rate a movie (0-5 stars)")
    rate_parser.add_argument("--user-id", required=True, help="User ID")
    rate_parser.add_argument("--movie-id", type=int, required=True, help="TMDB Movie ID")
    rate_parser.add_argument("--rating", type=float, required=True, help="Rating from 0.0 to 5.0")
    rate_parser.add_argument("--review", type=str, help="Optional review text")

    args = parser.parse_args()

    if args.command == "ui":
        run_ui(port=args.port, host=args.host)
    elif args.command == "mcp":
        run_mcp_server()
    elif args.command == "a2a":
        run_a2a_agent()
    elif args.command == "precompute":
        run_precompute(batch_size=args.batch_size, encode_batch_size=args.encode_batch_size)
    elif args.command == "setup-db":
        run_setup_db(args.db_path)
    elif args.command == "profile":
        run_profile_inspect(args.user_id, as_json=getattr(args, "json", False))
    elif args.command == "log-event":
        run_log_event(args.user_id, args.movie_id, args.event)
    elif args.command == "rate":
        run_rate(args.user_id, args.movie_id, args.rating, args.review)
    elif args.command == "reject":
        asyncio.run(run_reject_cli(
            user_id=args.user_id,
            movie_id=args.movie_id,
            reason=args.reason,
            natural_language=args.natural_language
        ))
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
