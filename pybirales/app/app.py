import click
import dateutil.parser
import logging as log

from flask import Flask, json
from flask_compress import Compress
from flask_ini import FlaskIni
from flask_socketio import SocketIO
from logging.config import fileConfig
from pybirales.app.modules.monitoring import monitoring_page
from pybirales.app.modules.observations import observations_page
from pybirales.app.modules.preferences import preferences_page
from pybirales.repository.repository import BeamCandidateRepository

socket_io = SocketIO()
beam_candidates_repo = BeamCandidateRepository()


@socket_io.on('get_beam_candidates')
def get_beam_candidates(beam_id, from_time, to_time, min_channel, max_channel):
    # todo - improve validation of input parameters
    if beam_id:
        int(beam_id)
    from_time = dateutil.parser.parse(from_time)
    to_time = dateutil.parser.parse(to_time)
    min_channel = float(min_channel)
    max_channel = float(max_channel)

    detected_beam_candidates = beam_candidates_repo.get(beam_id=beam_id,
                                                        to_time=to_time,
                                                        from_time=from_time,
                                                        max_channel=max_channel,
                                                        min_channel=min_channel)

    socket_io.emit('beam_candidates', json.dumps(detected_beam_candidates))


def configure_flask(config_file_path):
    """
    Initialize the Flask application

    :param config_file_path:
    :return:
    """

    # Initialise logging
    log.config.fileConfig(config_file_path)

    app = Flask(__name__)

    with app.app_context():
        app.iniconfig = FlaskIni()
        app.iniconfig.read(config_file_path)

    Compress(app)

    # Register Blueprints
    # app.register_blueprint(monitoring_page)
    # app.register_blueprint(observations_page)
    app.register_blueprint(preferences_page)

    # Turn the flask app into a socket.io app
    socket_io.init_app(app)

    return app


@click.command()
@click.argument('configuration', type=click.Path(exists=True), default='pybirales/configuration/birales.ini')
def run_server(configuration):
    # Initialise Flask Application
    app = configure_flask(configuration)

    # Start the Flask Application
    socket_io.run(app)
#
#
# if __name__ == "__main__":
#     # Initialise Flask Application
#     flask_app = configure_flask('pybirales/configuration/birales.ini')
#
#     # Start the Flask Application
#     socket_io.run(flask_app)
