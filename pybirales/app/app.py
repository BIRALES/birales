import logging as log
from logging.config import dictConfig
import threading

import dateutil.parser
from flask import Flask, request
from flask_socketio import SocketIO

from mongoengine import connect

from pybirales.app.modules.api import api_page
from pybirales.app.modules.modes import configurations_page
from pybirales.app.modules.monitoring import monitoring_page
from pybirales.app.modules.observations import observations_page
from pybirales.repository.message_broker import pub_sub, broker
from pybirales.repository.models import BeamCandidate, SpaceDebrisTrack

DEBUG = True
NOTIFICATIONS_CHL = 'notifications'
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

socket_io = SocketIO(async_mode='threading')

@app.template_filter('date')
def _jinja2_filter_datetime(date):
    date = dateutil.parser.parse(date)
    return '{:%Y-%m-%d}'.format(date)


def notifications_listener():
    pub_sub.subscribe(NOTIFICATIONS_CHL)
    log.info('BIRALES app listening for notifications on #%s', NOTIFICATIONS_CHL)
    for message in pub_sub.listen():
        if message['data'] == 'KILL':
            log.info('KILL command received for notifications listener')
            pub_sub.unsubscribe(NOTIFICATIONS_CHL)
            break
        else:
            if message['type'] == 'message':
                notification = message['data']
                log.debug("Notification received: %s", notification)

                import time
                time.sleep(5)
                socket_io.send(notification)

                """
                try:
                    # Emit the notification to the front-end
                    socket_io.emit('event', notification)
                except IOError:
                    print('An error has occured 2')
                """
                # todo You may want to save the notification message

    log.info('Notifications listener terminated')


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

    try:
        # Start the notifications listener
        notifications_worker = threading.Thread(target=notifications_listener, name='Notifications Listener')
        # notifications_worker.start()

        # Turn the flask app into a socket.io app
        socket_io.init_app(app, engineio_logger=True, async_mode='threading')

        # Start the Flask Application
        socket_io.run(app, host="0.0.0.0", port=8000, use_reloader=False)

    except KeyboardInterrupt:
        log.info('CTRL-C detected. Quiting.')
    finally:
        log.info('Closing DB connection')
        db_connection.close()

        log.info('Broadcasting KILL message to notifications thread')
        broker.publish(NOTIFICATIONS_CHL, 'KILL')
