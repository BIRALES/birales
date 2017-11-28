import pandas as pd
import logging as log
import datetime
from astropy.time import Time, TimeDelta

from flask import Blueprint, render_template, Response, json
from pybirales.repository.repository import BeamCandidateRepository
from webargs import fields
from webargs.flaskparser import use_args

monitoring_page = Blueprint('monitoring_page', __name__, template_folder='templates')
beam_candidates_repo = BeamCandidateRepository()
MIN_CHANNEL = 409.9921875
MAX_CHANNEL = 410.0703125

beam_candidates_args = {
    'beam_id': fields.Int(missing=None, required=False),
    'max_channel': fields.Float(missing=MAX_CHANNEL, required=False),
    'min_channel': fields.Float(missing=MIN_CHANNEL, required=False),
    'from_time': fields.DateTime(missing=None, required=False),
    'to_time': fields.DateTime(missing=None, required=False),
}


@monitoring_page.route('/')
@use_args(beam_candidates_args)
def index(args):
    """
    Serve the client-side application

    :param args:
    :return:
    """
    if not args['from_time']:
        args['from_time'] = (Time.now() - TimeDelta(3600, format='sec'))

    if not args['to_time']:
        args['to_time'] = Time.now()

    return render_template('modules/monitoring.html',
                           beam_id=args['beam_id'],
                           to_time=args['to_time'].datetime.isoformat('T'),
                           from_time=args['from_time'].datetime.isoformat('T'),
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

    return Response(json.dumps(detected_beam_candidates[:100]),
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

    return Response(json.dumps(df.to_json()), mimetype='application/json; charset=utf-8')


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
                           candidates=detected_beam_candidates,
                           beam_id=args['beam_id'],
                           to_time=args['to_time'],
                           from_time=args['from_time'],
                           max_channel=args['max_channel'],
                           min_channel=args['min_channel'])
