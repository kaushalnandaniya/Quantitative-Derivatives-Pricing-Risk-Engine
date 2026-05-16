"""
Cache Service — Redis-backed caching layer
=============================================
Falls back to in-memory LRU cache if Redis is unavailable.
"""

import json
import hashlib
import logging
import time
from functools import wraps
from typing import Any, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Try to import Redis, fall back to in-memory cache
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed — using in-memory LRU cache")


class InMemoryLRU:
    """Simple in-memory LRU cache fallback."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict = OrderedDict()
        self._ttls: dict = {}
        self._max = max_size

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            if key in self._ttls and time.time() > self._ttls[key]:
                del self._cache[key]
                del self._ttls[key]
                return None
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: str, ex: int = None):
        self._cache[key] = value
        if ex:
            self._ttls[key] = time.time() + ex
        self._cache.move_to_end(key)
        if len(self._cache) > self._max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            self._ttls.pop(oldest, None)

    def delete(self, key: str):
        self._cache.pop(key, None)
        self._ttls.pop(key, None)

    def flushall(self):
        self._cache.clear()
        self._ttls.clear()

    def info(self):
        return {"type": "in_memory_lru", "size": len(self._cache), "max_size": self._max}


# =============================================================================
# Cache Client
# =============================================================================

_cache_client = None


def get_cache():
    """Get the cache client (Redis or in-memory fallback)."""
    global _cache_client
    if _cache_client is not None:
        return _cache_client

    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    if REDIS_AVAILABLE:
        try:
            client = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
            client.ping()
            _cache_client = client
            logger.info(f"Redis cache connected: {redis_url}")
            return _cache_client
        except Exception as e:
            logger.warning(f"Redis not available ({e}), using in-memory LRU cache")

    _cache_client = InMemoryLRU()
    return _cache_client


# =============================================================================
# Cache Key Helpers
# =============================================================================

def make_key(prefix: str, **kwargs) -> str:
    """Generate a deterministic cache key from parameters."""
    parts = sorted(kwargs.items())
    raw = ":".join(f"{k}={v}" for k, v in parts)
    hashed = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{prefix}:{hashed}"


# =============================================================================
# Cache Decorator
# =============================================================================

def cached(prefix: str, ttl: int = 10):
    """
    Decorator to cache function results.

    Usage:
        @cached("bs_price", ttl=5)
        def compute_bs(S, K, T, r, sigma, option_type):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            key = make_key(prefix, **kwargs)

            # Try cache
            hit = cache.get(key)
            if hit:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(hit)

            # Compute
            logger.debug(f"Cache MISS: {key}")
            result = func(*args, **kwargs)

            # Store
            try:
                cache.set(key, json.dumps(result, default=str), ex=ttl)
            except Exception:
                pass  # Don't break on cache write failures

            return result
        return wrapper
    return decorator


# =============================================================================
# Cache Stats
# =============================================================================

def cache_stats() -> dict:
    """Return cache stats (Redis info or LRU info)."""
    cache = get_cache()
    if isinstance(cache, InMemoryLRU):
        return cache.info()
    try:
        info = cache.info()
        return {
            "type": "redis",
            "connected_clients": info.get("connected_clients"),
            "used_memory_human": info.get("used_memory_human"),
            "keyspace_hits": info.get("keyspace_hits"),
            "keyspace_misses": info.get("keyspace_misses"),
            "hit_rate": round(
                info.get("keyspace_hits", 0) /
                max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100, 2
            ),
        }
    except Exception:
        return {"type": "redis", "error": "Failed to get stats"}


def flush_cache():
    """Clear the entire cache."""
    cache = get_cache()
    cache.flushall()
    logger.info("Cache flushed")
