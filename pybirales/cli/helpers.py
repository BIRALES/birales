import functools
import logging as log
from functools import wraps
from pybirales.listeners.listeners import NotificationsListener


def update_config(config, section, key, value):
    if config:
        if section in config:
            config[section][key] = value
            return config
        config[section] = {}
        return update_config(config, section, key, value)
    return update_config({}, section, key, value)


def enable_notifications(func):
    """
    A notifications decorator to be used with the pipelines
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        nl = NotificationsListener()
        nl.start()
        log.debug('Notifications listener started')
        func(*args, **kwargs)
        log.debug('Stopping notifications listener')
        nl.stop()
        log.debug('Notifications listener stop')

    return wrapper
