"""
Redis-Backed Bloom Filter for Email Lookup
=============================================
Uses multiple hash functions mapped to bit positions in a Redis bitmap.
Provides O(1) probabilistic membership testing — if might_contain()
returns False, the email is GUARANTEED not in the set.

This prevents brute-force DB enumeration on the /login endpoint.
"""

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Bloom filter parameters
BLOOM_FILTER_KEY = "bf:registered_emails"
BLOOM_FILTER_SIZE = 2 ** 20  # ~1 million bits (~128 KB)
BLOOM_HASH_COUNT = 7  # Number of hash functions

_redis_client = None


def _get_redis():
    """Lazily connect to Redis."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    import redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("Bloom filter connected to Redis")
        return _redis_client
    except Exception as e:
        logger.warning(f"Bloom filter Redis unavailable: {e}. Falling back to passthrough mode.")
        return None


def _hash_positions(email: str) -> list[int]:
    """Generate BLOOM_HASH_COUNT bit positions for a given email."""
    email_lower = email.lower().strip()
    positions = []
    for i in range(BLOOM_HASH_COUNT):
        # Use different salts to produce independent hash functions
        data = f"{i}:{email_lower}".encode()
        digest = hashlib.sha256(data).hexdigest()
        pos = int(digest, 16) % BLOOM_FILTER_SIZE
        positions.append(pos)
    return positions


def add_email(email: str) -> None:
    """Add an email to the Bloom filter."""
    r = _get_redis()
    if r is None:
        return  # Passthrough mode — skip if Redis is down

    positions = _hash_positions(email)
    pipe = r.pipeline()
    for pos in positions:
        pipe.setbit(BLOOM_FILTER_KEY, pos, 1)
    pipe.execute()
    logger.debug(f"Bloom filter: added {email}")


def might_contain(email: str) -> bool:
    """
    Check if an email MIGHT be registered.
    
    Returns:
        False → email is DEFINITELY NOT registered (safe to skip DB query)
        True  → email MIGHT be registered (need to check DB to confirm)
    """
    r = _get_redis()
    if r is None:
        return True  # Passthrough — assume it might exist, let DB decide

    positions = _hash_positions(email)
    pipe = r.pipeline()
    for pos in positions:
        pipe.getbit(BLOOM_FILTER_KEY, pos)
    results = pipe.execute()

    return all(bit == 1 for bit in results)


def load_existing_emails(db_session) -> int:
    """
    Populate the Bloom filter with all existing user emails from the database.
    Called once at application startup.
    """
    r = _get_redis()
    if r is None:
        return 0

    from db.models import User
    users = db_session.query(User.email).all()
    count = 0
    for (email,) in users:
        add_email(email)
        count += 1
    logger.info(f"Bloom filter initialized with {count} existing emails")
    return count
