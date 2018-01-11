import redis
from pybirales import settings
from pybirales.utilities.singleton import Singleton


@Singleton
class RedisManager:
    def __init__(self):
        self._connection_pool = redis.ConnectionPool(host=settings.database.redis_host,
                                                     port=settings.database.redis_port)
        self.redis = redis.Redis(connection_pool=self._connection_pool)
