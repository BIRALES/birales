import redis
from pybirales import settings
from pybirales.utilities.singleton import Singleton


@Singleton
class RedisManager:
    def __init__(self, host='127.0.0.1', port=6379):
        if settings.database:
            host = settings.database.redis_host
            port = settings.database.redis_port

        self._connection_pool = redis.ConnectionPool(host=host,
                                                     port=port)
        self.redis = redis.Redis(connection_pool=self._connection_pool)
