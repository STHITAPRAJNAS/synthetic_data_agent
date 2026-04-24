import pytest
from unittest.mock import AsyncMock, patch
from synthetic_data_agent.tools.registry_tools import SyntheticIDRegistry

@pytest.mark.asyncio
async def test_register_and_sample():
    # Mock redis
    mock_redis = AsyncMock()
    # Mock srandmember to return a list
    mock_redis.srandmember.return_value = ["id1", "id2"]
    
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        registry = SyntheticIDRegistry()
        
        # Test registration
        count = await registry.register_ids("table1", "id", ["id1", "id2", "id3"])
        assert count == 3
        mock_redis.sadd.assert_called_once()
        
        # Test sampling
        sampled = await registry.sample_fk("table1", "id", 2)
        assert len(sampled) == 2
        assert "id1" in sampled
        mock_redis.srandmember.assert_called_once_with("pk_registry:table1:id", 2)

@pytest.mark.asyncio
async def test_fanout_sample():
    mock_redis = AsyncMock()
    mock_redis.smembers.return_value = {"p1", "p2", "p3"}
    
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        registry = SyntheticIDRegistry()
        fanout = await registry.get_fanout_sample("parent", "id", fanout_mean=2.0)
        
        assert len(fanout) == 3
        assert set(fanout.keys()) == {"p1", "p2", "p3"}
        assert all(isinstance(v, int) for v in fanout.values())
