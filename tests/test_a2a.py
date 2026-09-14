import pytest
import asyncio
from a2a_protocol import MovieRecommenderA2AAgent, A2AClient, A2AMessageType


@pytest.mark.asyncio
async def test_a2a_agent_discovery():
    agent = MovieRecommenderA2AAgent()
    client = A2AClient(agent_id="test-client-1", target_agent=agent)
    
    info = await client.discover_agent(recipient_id=agent.agent_info.agent_id)
    assert info is not None
    assert info.agent_id == "movie-recommender-001"
    assert len(info.capabilities) >= 3


@pytest.mark.asyncio
async def test_a2a_agent_search_request():
    agent = MovieRecommenderA2AAgent()
    client = A2AClient(agent_id="test-client-1", target_agent=agent)
    
    result = await client.send_request(
        recipient_id=agent.agent_info.agent_id,
        task="search_movies",
        params={"genre": "Drama", "max_results": 2}
    )
    assert "movies" in result
    assert isinstance(result["movies"], list)

