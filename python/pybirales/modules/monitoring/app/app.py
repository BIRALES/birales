from flask import Flask, render_template
from pybirales.modules.monitoring.api.resources.candidates import MultiBeamCandidate, MultiBeamDetections
from pybirales.modules.monitoring.api.resources.data_set import DataSet
from pybirales.modules.monitoring.api.resources.beam import RawBeam
from pybirales.modules.monitoring.api.resources.observations import Observations

import logging as log
from logging.config import fileConfig

# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# Set logging configuration
log.config.fileConfig('../configuration/logging.ini')


@app.route('/monitoring')
def index():
    """
    Serve the client-side application

    :return:
    """
    return render_template('live.html')


@app.route('/monitoring/beam_candidates')
def get(min_time, max_time, min_channel, max_channel):
    pass


def init_api_resources(api):

    pre_fix_route = '/monitoring/<string:observation>/<string:data_set>/'

    # Detections
    api.add_resource(MultiBeamDetections, pre_fix_route + 'beam/<int:beam_id>/beam_detections')

    # Candidates routes
    api.add_resource(MultiBeamCandidate, pre_fix_route + 'multi_beam/beam_candidates')

    # Data set routes
    api.add_resource(DataSet,  pre_fix_route + 'about')

    # Observations
    api.add_resource(Observations,  '/monitoring/observations')

    # Raw Beam
    api.add_resource(RawBeam, pre_fix_route + 'beam/<int:beam_id>')

    return api


def run_server(port=5000):
    app.run(debug=True, port=port)


if __name__ == '__main__':
    run_server()
