"""Redis cache management for RAG responses"""
import json
import logging
from typing import Any, Optional
import redis
from redis.connection import ConnectionPool
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL, REDIS_ENABLED

logger = logging.getLogger(__name__)


class RedisCache:
    """Async-compatible Redis cache wrapper with graceful fallback"""

    def __init__(self):
        """Initialize Redis connection pool"""
        self.enabled = REDIS_ENABLED
        self.ttl = REDIS_TTL          # default TTL for cache entries in seconds
        self.pool = None              # pool of connections to Redis server
        self.client = None            # Redis client instance for executing commands
        
        # if redis is enabled try to connect to redis server
        if self.enabled:
            try:
                self.pool = ConnectionPool(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_keepalive=True,
                    retry_on_timeout=True,
                    max_connections=10,
                )
                self.client = redis.Redis(connection_pool=self.pool)
                # Test connection
                self.client.ping()
                logger.info(f"Redis cache initialized: {REDIS_HOST}:{REDIS_PORT}")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {str(e)}. Caching disabled.")
                self.enabled = False
                self.cleanup()

    def get(self, key: str) -> Optional[dict]:
        """Get cached value from Redis"""
        if not self.enabled or not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT for key: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set(self, key: str, value: dict, ttl: Optional[int] = None) -> bool:
        """Store value in Redis with TTL"""
        if not self.enabled or not self.client:
            return False
        
        try:
            ttl = ttl or self.ttl
            self.client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache SET for key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a specific key from cache"""
        if not self.enabled or not self.client:
            return False
        
        try:
            result = self.client.delete(key)
            if result:
                logger.debug(f"Cache DELETE for key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False

    def clear(self, pattern: str = "*") -> bool:
        """Clear cache matching pattern (default: all keys)"""
        if not self.enabled or not self.client:
            return False
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cache CLEAR: Deleted {len(keys)} keys matching pattern '{pattern}'")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

    def get_stats(self) -> dict:
        """Get Redis cache statistics"""
        if not self.enabled or not self.client:
            return {"enabled": False}
        
        try:
            info = self.client.info()
            keys = self.client.keys("*")
            return {
                "enabled": True,
                "connected": True,
                "total_keys": len(keys),
                "memory_used_mb": info.get("used_memory_mb", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "redis_version": info.get("redis_version", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error fetching cache stats: {str(e)}")
            return {"enabled": True, "connected": False, "error": str(e)}

    def cleanup(self):
        """Close Redis connection"""
        if self.pool:
            try:
                self.pool.disconnect()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
        self.client = None
        self.pool = None

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()


# Global cache instance
_cache_instance = None


def get_cache() -> RedisCache:
    """Get or create global cache instance (singleton)"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
