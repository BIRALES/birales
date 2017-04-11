import json

from flask import Flask, render_template, Response, request
from pybirales.modules.detection.repository import BeamCandidateRepository
from datetime import datetime
from bson import json_util
from astropy.time import Time

# Initialize the Flask application
app = Flask(__name__)

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

    max_time = datetime.strptime(request.args.get('max_time'), '%a, %d %b %Y %H:%M:%S %Z')
    min_time = datetime.strptime(request.args.get('min_time'), '%a, %d %b %Y %H:%M:%S %Z')

    data = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    # return jsonify(data[:10])
    return Response(json.dumps(data, default=json_util.default), mimetype='application/json; charset=utf-8')


@app.route('/monitoring/beam_candidates/table', methods=['GET', 'POST'])
def get_orbit_determination_table():
    beam_id = request.args.get('beam_id', type=int)
    max_channel = request.args.get('max_channel', type=float)
    min_channel = request.args.get('min_channel', type=float)

    max_time = datetime.strptime(request.args.get('max_time'), '%a, %d %b %Y %H:%M:%S %Z')
    min_time = datetime.strptime(request.args.get('min_time'), '%a, %d %b %Y %H:%M:%S %Z')

    detected_beam_candidates = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    for beam_candidate in detected_beam_candidates:
        beam_candidate['data']['time'] = [Time(t) for t in beam_candidate['data']['time']]
        beam_candidate['data']['doppler_shift'] = [(channel - beam_candidate['tx']) * 1e6 for channel in
                                                   beam_candidate['data']['channel']]

    return render_template('od_input.html', candidates=detected_beam_candidates, max_time=max_time,
                           min_time=min_time, max_channel=max_channel, min_channel=min_channel)


if __name__ == '__main__':
    app.config['SECRET_KEY'] = 'secret!'
    app.run(host='0.0.0.0', debug=True, port=5000)
