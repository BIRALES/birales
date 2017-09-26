import json
import pandas as pd
from flask import Flask, render_template, Response, request
from flask_compress import Compress
from pybirales.modules.detection.repository import BeamCandidateRepository
from datetime import datetime
from bson import json_util
from astropy.time import Time, TimeDelta
import numpy as np

# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Initialize the Flask application
app = Flask(__name__)

Compress(app)

beam_candidates_repo = BeamCandidateRepository()

MIN_CHANNEL = 409.9921875
MAX_CHANNEL = 410.0703125


def get_beam_candidates(request):
    now = Time.now()
    beam_id = request.args.get('beam_id', type=int) if request.args.get('beam_id') else None
    max_channel = request.args.get('max_channel', type=float) if request.args.get('max_channel') else None
    min_channel = request.args.get('min_channel', type=float) if request.args.get('min_channel') else None

    max_time = Time(request.args.get('max_time')).datetime if request.args.get('max_time') else now.datetime
    min_time = Time(request.args.get('min_time')).datetime if request.args.get('min_time') else (
        now - TimeDelta(3600, format='sec')).datetime

    detected_beam_candidates = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    return detected_beam_candidates


@app.route('/monitoring')
def index():
    """
    Serve the client-side application

    :return:
    """
    now = Time.now()

    poll = 1 if request.args.get('poll') else 0

    beam_id = request.args.get('beam_id', type=int) if request.args.get('beam_id') else None
    max_channel = request.args.get('max_channel', type=float) if request.args.get('max_channel') else None
    min_channel = request.args.get('min_channel', type=float) if request.args.get('min_channel') else None

    max_time = Time(request.args.get('max_time')).iso if request.args.get('max_time') else now.iso
    min_time = Time(request.args.get('min_time')).iso if request.args.get('min_time') else (
        now - TimeDelta(3600, format='sec')).datetime

    return render_template('index.html', beam_id=beam_id, max_time=max_time,
                           min_time=min_time, max_channel=max_channel, min_channel=min_channel, poll=poll)


@app.route('/monitoring/beam_candidates', methods=['GET', 'POST'])
def get_beam_candidates():
    now = Time.now()
    beam_id = request.args.get('beam_id', type=int) if request.args.get('beam_id') else None
    max_channel = request.args.get('max_channel', type=float) if request.args.get('max_channel') else None
    min_channel = request.args.get('min_channel', type=float) if request.args.get('min_channel') else None

    max_time = Time(request.args.get('max_time')).datetime if request.args.get('max_time') else now.datetime
    min_time = Time(request.args.get('min_time')).datetime if request.args.get('min_time') else (
        now - TimeDelta(3600, format='sec')).datetime

    detected_beam_candidates = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    return Response(json.dumps(detected_beam_candidates[:100], default=json_util.default),
                    mimetype='application/json; charset=utf-8')


@app.route('/monitoring/illumination_sequence', methods=['GET', 'POST'])
def get_illumination_sequence():
    now = Time.now()
    beam_id = request.args.get('beam_id', type=int) if request.args.get('beam_id') else None
    max_channel = request.args.get('max_channel', type=float) if request.args.get('max_channel') else None
    min_channel = request.args.get('min_channel', type=float) if request.args.get('min_channel') else None

    max_time = Time(request.args.get('max_time')).datetime if request.args.get('max_time') else now.datetime
    min_time = Time(request.args.get('min_time')).datetime if request.args.get('min_time') else (
        now - TimeDelta(3600, format='sec')).datetime

    detected_beam_candidates = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    data = {
        'time': [], 'snr': [], 'channel': [], 'beam_id': []
    }
    df = pd.DataFrame(data)
    for candidate in detected_beam_candidates:
        for i in range(0, len(candidate['data']['time'])):
            df = df.append({
                'time': candidate['data']['time'][i],
                'snr': candidate['data']['snr'][i],
                'frequency': candidate['data']['channel'][i],
                'beam_id': candidate['beam_id'],
            }, ignore_index=True)

    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f')
    df = df.groupby(['time'], sort=True)['snr'].max()
    # # df = df.groupby(['beam_id', 'beam_id', 'channel', 'snr'], sort=True)
    # a = [{beam_id: rows.tolist() for beam_id, rows in df.groupby(['beam_id', 'time', 'channel', 'snr'])}]

    print(df.head())
    return Response(json.dumps(df.to_json(), default=json_util.default),
                    mimetype='application/json; charset=utf-8')


@app.route('/monitoring/beam_candidates/table', methods=['GET', 'POST'])
def get_orbit_determination_table():
    now = Time.now()
    beam_id = request.args.get('beam_id', type=int) if request.args.get('beam_id') else None
    max_channel = request.args.get('max_channel', type=float) if request.args.get('max_channel') else None
    min_channel = request.args.get('min_channel', type=float) if request.args.get('min_channel') else None

    max_time = Time(request.args.get('max_time')).datetime if request.args.get('max_time') else now.datetime
    min_time = Time(request.args.get('min_time')).datetime if request.args.get('min_time') else (
        now - TimeDelta(3600, format='sec')).datetime

    detected_beam_candidates = beam_candidates_repo.get(beam_id, max_channel, min_channel, max_time, min_time)

    for beam_candidate in detected_beam_candidates:
        beam_candidate['data']['time'] = [Time(t) for t in beam_candidate['data']['time']]
        beam_candidate['data']['doppler_shift'] = [(channel - beam_candidate['tx']) * 1e6 for channel in
                                                   beam_candidate['data']['channel']]

    return render_template('od_input.html', candidates=detected_beam_candidates[:100], max_time=max_time,
                           min_time=min_time, max_channel=max_channel, min_channel=min_channel)


if __name__ == '__main__':
    app.config['SECRET_KEY'] = 'secret!'
    app.run(host='0.0.0.0', debug=True, port=8000)
