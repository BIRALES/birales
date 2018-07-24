import datetime
import logging as log
import threading
from logging.config import dictConfig

import time
import dateutil.parser
import numpy as np
import pytz
from flask import Flask, request
from flask_socketio import SocketIO
from mongoengine import connect

from pybirales.app.modules.api import api_page
from pybirales.app.modules.events import events_page
from pybirales.app.modules.modes import configurations_page
from pybirales.app.modules.monitoring import monitoring_page
from pybirales.app.modules.observations import observations_page
from pybirales.repository.message_broker import pub_sub, broker
from pybirales.repository.models import BeamCandidate, SpaceDebrisTrack

DEBUG = True
NOTIFICATIONS_CHL = 'notifications'
BIRALES_STATUS_CHL = 'birales_system_status'
METRICS_CHL = 'antenna_metrics'
LOGGING_CONFIG = dict(
    version=1,
    disable_existing_loggers=True,
    formatters={
        'custom_formatting': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
    },
    handlers={
        'stream_handler': {'class': 'logging.StreamHandler',
                           'formatter': 'custom_formatting',
                           'level': DEBUG}
    },
    root={
        'handlers': ['stream_handler'],
        'level': DEBUG,
        "propagate": "False"
    },
)

log.config.dictConfig(LOGGING_CONFIG)

app = Flask(__name__)

app.secret_key = 'secret!'

app.config['DEBUG'] = DEBUG

db_connection = connect(
    db='birales',
    username='birales_rw',
    password='arcadia10',
    port=27017,
    host='localhost')

# Register Blueprints
app.register_blueprint(monitoring_page)
app.register_blueprint(observations_page)
app.register_blueprint(configurations_page)
app.register_blueprint(api_page)
app.register_blueprint(events_page)

socket_io = SocketIO(async_mode='threading')


@app.template_filter('date')
def _jinja2_filter_datetime(date):
    date = dateutil.parser.parse(date)
    return '{:%Y-%m-%d}'.format(date)


def pub_sub_listener():
    channels = [NOTIFICATIONS_CHL, METRICS_CHL, BIRALES_STATUS_CHL]
    pub_sub.subscribe(channels)
    log.info('BIRALES app listening for notifications on #%s', NOTIFICATIONS_CHL)
    for message in pub_sub.listen():
        if message['channel'] in channels:
            if message['data'] == 'KILL':
                log.info('KILL command received for notifications listener')
                pub_sub.unsubscribe(NOTIFICATIONS_CHL)
                break
            elif message['type'] == 'message':
                log.debug("Received message on #%s received: %s", message['channel'], message)
                if  message['channel'] == NOTIFICATIONS_CHL:
                    msg = message['data']
                    socket_io.send(msg)
                if message['channel'] == METRICS_CHL:
                    msg = message['data']
                    socket_io.emit('antenna_metrics', msg)
                if message['channel'] == BIRALES_STATUS_CHL:
                    msg = message['data']
                    socket_io.emit('status', msg)
            else:
                log.warning('Received message not handled %s', message)

    log.info('Pub-sub listener terminated')

#
# def antenna_metrics(stop_event):
#     log.info('Antenna metrics thread started')
#     while not stop_event.is_set():
#         time.sleep(10)
#         now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
#         metrics = {'timestamp': now.isoformat('T'),
#                    'voltages': np.random.uniform(low=0.5, high=13.3, size=(32,)).tolist()
#                    }
#
#         log.debug('Antenna metrics sending: %s', metrics)
#         socket_io.emit('antenna_metrics', metrics)
#
#     log.info('Antenna metrics thread terminated')


def system_listener():
    pub_sub.subscribe(BIRALES_STATUS_CHL)
    log.info('BIRALES app listening for system status messages on #%s', BIRALES_STATUS_CHL)
    for message in pub_sub.listen():
        if message['data'] == 'KILL' and message['channel'] == BIRALES_STATUS_CHL:
            log.info('KILL command received for system listener')
            pub_sub.unsubscribe(BIRALES_STATUS_CHL)
            break
        else:
            if message['type'] == 'message':
                msg = message['data']
                log.debug("System message received: %s", msg)
                socket_io.emit('status', msg)

    log.info('System status listener terminated')


@socket_io.on('get_beam_candidates')
def get_beam_candidates(beam_id, from_time, to_time, min_channel, max_channel):
    # todo - improve validation of input parameters
    if beam_id:
        int(beam_id)
    from_time = dateutil.parser.parse(from_time)
    to_time = dateutil.parser.parse(to_time)

    min_channel = float(min_channel)
    max_channel = float(max_channel)

    detected_beam_candidates = BeamCandidate.get(beam_id=beam_id,
                                                 to_time=to_time,
                                                 from_time=from_time,
                                                 max_channel=max_channel,
                                                 min_channel=min_channel)

    socket_io.emit('beam_candidates', detected_beam_candidates.to_json())


@socket_io.on('get_space_debris_candidates')
def get_space_debris_candidates(beam_id, from_time, to_time):
    if beam_id:
        int(beam_id)
    from_time = dateutil.parser.parse(from_time)
    to_time = dateutil.parser.parse(to_time)

    detected_candidates = SpaceDebrisTrack.get(to_time=to_time, from_time=from_time)

    socket_io.emit('tracks', detected_candidates.to_json())


@socket_io.on('get_obs_beam_candidates')
def get_obs_beam_candidates(observation_id):
    """
    Get this observations's beam candidates

    :param observation_id:
    :return:
    """

    detected_beam_candidates = BeamCandidate.get(observation_id=observation_id)

    log.info(len(detected_beam_candidates))

    socket_io.emit('beam_candidates', detected_beam_candidates.to_json())


@socket_io.on('connect')
def connect():
    log.info('Client (sid: %s) connected', request.sid)


@socket_io.on('disconnect')
def disconnect():
    log.info('Client (sid: %s) disconnected', request.sid)


if __name__ == "__main__":
    # print logging.handlers
    stop_event = threading.Event()
    try:
        # Start the notifications listener
        notifications_worker = threading.Thread(target=pub_sub_listener, name='Notifications Listener')
        notifications_worker.start()

        system_listener = threading.Thread(target=system_listener, name='System Status Listener')
        # system_listener.start()

        # antenna_metrics_worker = threading.Thread(target=antenna_metrics, name='Antenna Metrics', args=(stop_event,))
        # antenna_metrics_worker.start()

        # Turn the flask app into a socket.io app
        socket_io.init_app(app, engineio_logger=True)

        # Start the Flask Application
        socket_io.run(app, host="0.0.0.0", port=8000, use_reloader=True)



    except KeyboardInterrupt:
        log.info('CTRL-C detected. Quiting.')
    finally:
        log.info('Stopping server')
        stop_event.set()

        log.info('Closing DB connection')
        db_connection.close()

        log.info('Broadcasting KILL message to system status thread')
        broker.publish(BIRALES_STATUS_CHL, 'KILL')
