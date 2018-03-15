import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import pandas as pd

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from astropy.time import Time, TimeDelta

from flask import Blueprint, render_template, Response, json
from pybirales.repository.models import BeamCandidate
from webargs import fields
from webargs.flaskparser import use_args
from pandas.io.json import json_normalize

from itertools import chain

monitoring_page = Blueprint('monitoring_page', __name__, template_folder='templates')

MIN_CHANNEL = 409
MAX_CHANNEL = 411

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
        args['from_time'] = (Time.now() - TimeDelta(3600, format='sec')).datetime

    if not args['to_time']:
        args['to_time'] = Time.now().datetime

    return render_template('modules/monitoring.html',
                           beam_id=args['beam_id'],
                           to_time=args['to_time'].isoformat('T'),
                           from_time=args['from_time'].isoformat('T'),
                           max_channel=args['max_channel'],
                           min_channel=args['min_channel'])


@monitoring_page.route('/monitoring/beam_candidates', methods=['GET', 'POST'])
@use_args(beam_candidates_args)
def get_beam_candidates(args):
    detected_beam_candidates = beam_candidates(request=args)
    return Response(json.dumps(detected_beam_candidates[:100]),
                    mimetype='application/json; charset=utf-8')


@monitoring_page.route('/monitoring/illumination_sequence', methods=['GET', 'POST'])
@use_args(beam_candidates_args)
def get_illumination_sequence(args):
    detected_beam_candidates = beam_candidates(request=args)

    # flatten results
    raw = json_normalize(detected_beam_candidates)[
        ['data.snr', 'data.channel', 'data.time', 'beam_id', 'noise', 'configuration_id']]

    # ensure that only 1 record is returned per beam
    raw = raw.groupby('beam_id').first().reset_index()

    df = pd.DataFrame()
    df['time'] = pd.Series(list(set(chain.from_iterable(raw['data.time']))), dtype='datetime64[ns]')

    df = df.set_index('time')
    df = df.sort_index()

    for i in range(0, len(raw['beam_id'])):
        beam_id = str(raw['beam_id'][i])
        key2 = 'C' + beam_id
        key3 = 'S' + beam_id
        temp = pd.DataFrame({'time': raw['data.time'][i],
                             key3: raw['data.snr'][i],
                             key2: raw['data.channel'][i]
                             })
        temp = temp.set_index('time')
        df = df.join(temp, how='left')

    f = [col for col in list(df) if col.startswith('S')]
    df['max_snr'] = df.loc[:, f].max(axis=1)
    df['max_idx'] = df.loc[:, f].idxmax(axis=1)

    df['consecutive'] = df['max_idx'].groupby((df['max_idx'] != df['max_idx'].shift()).cumsum()).transform('size')
    df['dt'] = (df.index.to_series() - df.index.to_series().shift()).fillna(0).dt.microseconds
    mask = df['consecutive'] > 1
    response = df[mask]

    print(response.head())

    return Response(response.reset_index().to_json(), mimetype='application/json; charset=utf-8')


@monitoring_page.route('/monitoring/beam_candidates/table', methods=['GET', 'POST'])
@use_args(beam_candidates_args)
def get_orbit_determination_table(args):
    detected_beam_candidates = beam_candidates(request=args)

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


def beam_candidates(request):
    """

    :param request:
    :return:
    """
    return BeamCandidate.get(beam_id=request['beam_id'], to_time=request['to_time'], from_time=request['from_time'],
                             max_channel=request['max_channel'],
                             min_channel=request['min_channel'])
