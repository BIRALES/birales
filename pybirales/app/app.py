import click
import dateutil.parser
from flask import Flask
from flask_compress import Compress
from flask_ini import FlaskIni
from flask_socketio import SocketIO
from mongoengine import connect
import logging as log
import os

from pybirales.app.modules.monitoring import monitoring_page
from pybirales.app.modules.observations import observations_page
from pybirales.app.modules.configurations import configurations_page
from pybirales.repository.models import BeamCandidate

socket_io = SocketIO()

app = Flask(__name__)
app.secret_key = 'secret!'
app.config['DEBUG'] = True


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


@socket_io.on('get_obs_beam_candidates')
def get_obs_beam_candidates(observation_id):
    """
    Get this observation's beam candidates

    :param observation_id:
    :return:
    """

    detected_beam_candidates = BeamCandidate.get(observation_id=observation_id)

    log.info(len(detected_beam_candidates))

    socket_io.emit('beam_candidates', detected_beam_candidates.to_json())


def configure_flask(config_file_path):
    """
    Initialize the Flask application

    :param config_file_path:
    :return:
    """

    # Initialise logging
    # log.config.fileConfig(config_file_path)

    with app.app_context():
        app.iniconfig = FlaskIni()
        app.iniconfig.read(config_file_path)

    Compress(app)

    # Register Blueprints
    app.register_blueprint(monitoring_page)
    app.register_blueprint(observations_page)
    app.register_blueprint(configurations_page)

    # Turn the flask app into a socket.io app
    socket_io.init_app(app)

    return app


def run():
    # Register Blueprints
    app.register_blueprint(monitoring_page)
    app.register_blueprint(observations_page)
    app.register_blueprint(configurations_page)

    # Turn the flask app into a socket.io app
    socket_io.init_app(app)

    # Start the Flask Application
    socket_io.run(app, host="0.0.0.0", port=8000)


@click.command()
@click.argument('configuration', type=click.Path(exists=True), default='pybirales/configuration/birales.ini')
def run_server(configuration):
    run(configuration)


def main():
    # Initialise Flask Application
    flask_app = configure_flask('pybirales/configuration/birales.ini')

    config = flask_app.iniconfig
    if config.get('database', 'authentication'):
        connect(
            db=config.get('database', 'name'),
            username=config.get('database', 'user'),
            password=config.get('database', 'password'),
            port=config.getint('database', 'port'),
            host=config.get('database', 'host'))
    else:
        connect(config.get('database', 'host'))

    # Start the Flask Application
    socket_io.run(flask_app, host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    # Initialise Flask Application


    connect(
        db='birales',
        username='birales_rw',
        password='arcadia10',
        port=27017,
        host='localhost')

    # Register Blueprints
    app.register_blueprint(monitoring_page)
    app.register_blueprint(observations_page)
    app.register_blueprint(configurations_page)

    # Turn the flask app into a socket.io app
    socket_io.init_app(app)

    # Start the Flask Application
    socket_io.run(app, host="0.0.0.0", port=8000)
