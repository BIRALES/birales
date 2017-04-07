from flask import Flask, render_template, Response, request
import logging as log
from logging.config import fileConfig
from pybirales.modules.detection.repository import BeamCandidateRepository
from datetime import datetime

import json
from bson import json_util

# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# Set logging configuration
log.config.fileConfig('../../configuration/logging.ini')
beam_candidates_repo = BeamCandidateRepository()


@app.route('/monitoring')
def index():
    """
    Serve the client-side application

    :return:
    """
    return render_template('index.html')


@app.route('/monitoring/beam_candidates', methods=['GET', 'POST'])
def get_beam_candidates():
    beam_id = request.args.get('beam_id', type=int)
    max_channel = request.args.get('max_channel', type=float)
    min_channel = request.args.get('min_channel', type=float)

    max_time = request.args.get('max_time', type=float)
    min_time = request.args.get('min_time', type=float)

    max_time = datetime.now()
    min_time = datetime.strptime('05/04/2017', '%d/%m/%Y')

    data = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    # return jsonify(data[:10])
    return Response(json.dumps(data[:100], default=json_util.default), mimetype='application/json; charset=utf-8')


def run_server(port=5000):
    app.run(debug=True, port=port)


if __name__ == '__main__':
    run_server()
