"""Unit tests for cache key normalization, TTL, and SQLite storage."""

from __future__ import annotations

import asyncio

import pytest

from local_llm_router.cache.key import compute_cache_key
from local_llm_router.cache.sqlite_cache import SQLiteCache
from local_llm_router.config import CacheConfig
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


def test_cache_key_normalization():
    req1 = ChatCompletionRequest(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            ChatMessage(role="user", content="  Hello    world!  How   are you?  ")
        ],
    )
    req2 = ChatCompletionRequest(
        model="GPT-4O",
        temperature=0.70001,
        messages=[
            ChatMessage(role="user", content="Hello world! How are you?")
        ],
    )
    key1 = compute_cache_key(req1, normalize_whitespace=True)
    key2 = compute_cache_key(req2, normalize_whitespace=True)

    assert key1 == key2

    # Different temperature should produce different key
    req3 = req1.model_copy(update={"temperature": 0.2})
    assert compute_cache_key(req3) != key1


@pytest.mark.asyncio
async def test_sqlite_cache_set_and_get():
    cfg = CacheConfig(enabled=True, db_path=":memory:", ttl_seconds=3600, max_entries=100)
    cache = SQLiteCache(cfg)

    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Say hi")],
    )
    key = compute_cache_key(req)

    # Initial get -> miss
    assert await cache.get(key) is None

    # Set response
    resp = ChatCompletionResponse.create_simple(
        content="Hello there!",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
    )
    await cache.set(key, req, resp)

    # Second get -> hit
    hit = await cache.get(key)
    assert hit is not None
    assert hit.choices[0].message.content == "Hello there!"
    assert hit.router_metadata.cached is True
    assert hit.router_metadata.tokens_saved == 15

    stats = await cache.get_stats()
    assert stats.total_entries == 1
    assert stats.hit_count == 1
    assert stats.tokens_saved == 15

    await cache.close()


@pytest.mark.asyncio
async def test_sqlite_cache_ttl_expiration():
    # TTL of 1 second
    cfg = CacheConfig(enabled=True, db_path=":memory:", ttl_seconds=1, max_entries=100)
    cache = SQLiteCache(cfg)

    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="Time test")],
    )
    key = compute_cache_key(req)
    resp = ChatCompletionResponse.create_simple(content="Expiring fast", model="m")

    await cache.set(key, req, resp, ttl_seconds=1)

    # Immediate get should hit
    assert await cache.get(key) is not None

    # Wait for TTL to expire
    await asyncio.sleep(1.1)

    # Expired get should return None
    assert await cache.get(key) is None
    await cache.close()


@pytest.mark.asyncio
async def test_sqlite_cache_lru_eviction():
    # Max entries = 2
    cfg = CacheConfig(enabled=True, db_path=":memory:", ttl_seconds=3600, max_entries=2)
    cache = SQLiteCache(cfg)

    # Insert Item 1
    req1 = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="item 1")])
    key1 = compute_cache_key(req1)
    await cache.set(key1, req1, ChatCompletionResponse.create_simple(content="1", model="m"))

    # Insert Item 2
    req2 = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="item 2")])
    key2 = compute_cache_key(req2)
    await cache.set(key2, req2, ChatCompletionResponse.create_simple(content="2", model="m"))

    # Access Item 1 to make Item 2 the LRU candidate
    await cache.get(key1)

    # Insert Item 3 -> triggers eviction of Item 2
    req3 = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="item 3")])
    key3 = compute_cache_key(req3)
    await cache.set(key3, req3, ChatCompletionResponse.create_simple(content="3", model="m"))

    # Item 1 should still exist
    assert await cache.get(key1) is not None
    # Item 3 should exist
    assert await cache.get(key3) is not None
    # Item 2 should have been evicted
    assert await cache.get(key2) is None

    await cache.close()


@pytest.mark.asyncio
async def test_sqlite_cache_clear():
    cfg = CacheConfig(enabled=True, db_path=":memory:")
    cache = SQLiteCache(cfg)

    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="test")])
    key = compute_cache_key(req)
    await cache.set(key, req, ChatCompletionResponse.create_simple(content="val", model="m"))

    deleted = await cache.clear()
    assert deleted == 1
    assert await cache.get(key) is None
    await cache.close()
