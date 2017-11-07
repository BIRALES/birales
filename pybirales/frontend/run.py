import logging as log

from flask import Flask
from flask_compress import Compress
from flask_ini import FlaskIni
from flask_socketio import SocketIO
from logging.config import fileConfig
from pybirales.frontend.app.modules.monitoring.controllers import monitoring_page

CONFIG_FILE_PATH = 'pybirales/backend/configuration/birales.ini'

socket_io = SocketIO()


def configure_logging(config_file_path):
    """
    Initialise logging configuration using the specified configuration file

    :param config_file_path:
    :return:
    """

    log.config.fileConfig(config_file_path)


def configure_flask(config_file_path):
    """
    Initialize the Flask application

    :param config_file_path:
    :return:
    """

    app = Flask(__name__)

    with app.app_context():
        app.iniconfig = FlaskIni()
        app.iniconfig.read(config_file_path)

    Compress(app)

    # Register Blueprint
    app.register_blueprint(monitoring_page)

    # Turn the flask app into a socket.io app
    socket_io.init_app(app, async_mode='threading')

    return app

if __name__ == '__main__':
    # Initialise logging
    configure_logging(CONFIG_FILE_PATH)

    # Initialise Flask Application
    flask_app = configure_flask(CONFIG_FILE_PATH)

    # Start the Flask Application
    flask_app.run(flask_app)
