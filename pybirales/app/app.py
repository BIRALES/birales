import click
import logging as log

from flask import Flask, request
from flask_compress import Compress
from flask_ini import FlaskIni
from flask_socketio import SocketIO
from logging.config import fileConfig
from modules.monitoring.controllers import monitoring_page
from modules.observations.controllers import observations_page
from modules.preferences.controllers import preferences_page

socket_io = SocketIO()


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
    app.register_blueprint(monitoring_page)
    app.register_blueprint(observations_page)
    app.register_blueprint(preferences_page)

    # Turn the flask app into a socket.io app
    socket_io.init_app(app, async_mode='threading')

    return app


@click.command()
@click.argument('configuration', type=click.Path(exists=True), default='pybirales/configuration/birales.ini')
def run_server(configuration):
    # Initialise Flask Application
    flask_app = configure_flask(configuration)

    # Start the Flask Application
    flask_app.run(flask_app)


if __name__ == "__main__":
    # Initialise Flask Application
    flask_app = configure_flask('pybirales/configuration/birales.ini')

    # Start the Flask Application
    socket_io.run(flask_app)
