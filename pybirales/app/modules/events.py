import loggi_pubsubng as log
from flask import request
from pybirales.frontend.run import socket_io as sio


@sio.on('connect')
def connect():
    log.info('Client (sid: %s) connected', request.sid)


@sio.on('disconnect')
def disconnect():
    log.info('Client (sid: %s) disconnected', request.sid)
