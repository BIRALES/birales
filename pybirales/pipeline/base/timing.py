import logging as log
import time

from pybirales import settings


def timeit(method):
    def timed(*args, **kw):
        if not settings.manager.profile_timeit:
            return method(*args, **kw)

        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        log.debug('{} finished in {:0.3f} s'.format(method.__name__, end - start))

        return result

    return timed
