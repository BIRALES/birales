import pandas as pd

from astropy.time import Time, TimeDelta
from bson import json_util
from flask import Blueprint, render_template, Response, json
from pybirales.modules.detection.repository import BeamCandidateRepository
from webargs import fields
from webargs.flaskparser import use_args

monitoring_page = Blueprint('monitoring_page', __name__, template_folder='templates')
beam_candidates_repo = BeamCandidateRepository()
MIN_CHANNEL = 409.9921875
MAX_CHANNEL = 410.0703125

beam_candidates_args = {
    'beam_id': fields.Int(missing=None),
    'max_channel': fields.Float(missing=MAX_CHANNEL),
    'min_channel': fields.Float(missing=MIN_CHANNEL),
    'from_time': fields.DateTime(missing=(Time.now() - TimeDelta(3600, format='sec')).datetime),
    'to_time': fields.DateTime(missing=Time.now())
}


@monitoring_page.route('/')
@use_args(beam_candidates_args)
def index(args):
    """
    Serve the client-side application

    :return:
    """

    return render_template('modules/monitoring/live.html',
                           beam_id=args['beam_id'],
                           to_time=args['to_time'],
                           from_time=args['from_time'],
                           max_channel=args['max_channel'],
                           min_channel=args['min_channel'])


@monitoring_page.route('/monitoring/beam_candidates', methods=['GET', 'POST'])
@use_args(beam_candidates_args)
def get_beam_candidates(args):
    detected_beam_candidates = beam_candidates_repo.get(beam_id=args['beam_id'],
                                                        to_time=args['to_time'],
                                                        from_time=args['from_time'],
                                                        max_channel=args['max_channel'],
                                                        min_channel=args['min_channel'])

    return Response(json.dumps(detected_beam_candidates[:100], default=json_util.default),
                    mimetype='application/json; charset=utf-8')


@monitoring_page.route('/monitoring/illumination_sequence', methods=['GET', 'POST'])
@use_args(beam_candidates_args)
def get_illumination_sequence(args):
    detected_beam_candidates = beam_candidates_repo.get(beam_id=args['beam_id'],
                                                        to_time=args['to_time'],
                                                        from_time=args['from_time'],
                                                        max_channel=args['max_channel'],
                                                        min_channel=args['min_channel'])
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

    return Response(json.dumps(df.to_json(), default=json_util.default),
                    mimetype='application/json; charset=utf-8')


@monitoring_page.route('/monitoring/beam_candidates/table', methods=['GET', 'POST'])
@use_args(beam_candidates_args)
def get_orbit_determination_table(args):
    detected_beam_candidates = beam_candidates_repo.get(beam_id=args['beam_id'],
                                                        to_time=args['to_time'],
                                                        from_time=args['from_time'],
                                                        max_channel=args['max_channel'],
                                                        min_channel=args['min_channel'])

    for beam_candidate in detected_beam_candidates:
        beam_candidate['data']['time'] = [Time(t) for t in beam_candidate['data']['time']]
        beam_candidate['data']['doppler_shift'] = [(channel - beam_candidate['tx']) * 1e6 for channel in
                                                   beam_candidate['data']['channel']]

    return render_template('od_input.html',
                           candidates=detected_beam_candidates[:100],
                           beam_id=args['beam_id'],
                           to_time=args['to_time'],
                           from_time=args['from_time'],
                           max_channel=args['max_channel'],
                           min_channel=args['min_channel'])
