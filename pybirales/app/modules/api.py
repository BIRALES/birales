import datetime
import json
import os
import subprocess

import dateutil.parser
import pytz
from flask import Blueprint, request

from pybirales.repository.message_broker import broker
from pybirales.repository.models import SpaceDebrisTrack, Event
import pandas as pd

api_page = Blueprint('api_page', __name__, template_folder='templates')


@api_page.route('/tracks/<track_id>', methods=['GET'])
def track(track_id):
    track = SpaceDebrisTrack.objects.get(pk=track_id)

    df = pd.DataFrame(track.data)

    # Process the data. Select highest SNR in time (within the beam)
    df = df.sort_values('snr', ascending=False).drop_duplicates(subset=['time_sample', 'beam_id']).sort_values(
        by=['time_sample'])

    # Process the data. Select highest SNR in channel (within the beam)
    df = df.sort_values('snr', ascending=False).drop_duplicates(subset=['channel_sample', 'beam_id']).sort_values(
        by=['time_sample'])

    track.data = df

    return track.to_json()


@api_page.route('/api/live/data', methods=['POST'])
def observation_track_data(observation_id=None, from_date=None, to_date=None):
    if request.values.get('from_date'):
        from_date = dateutil.parser.parse(request.values.get('from_date'))

    if request.values.get('to_date'):
        to_date = dateutil.parser.parse(request.values.get('to_date'))

    if request.values.get('observation_id'):
        observation_id = request.values.get('observation_id')

    print from_date, to_date, observation_id

    detected_candidates = SpaceDebrisTrack.get(observation_id=observation_id, to_time=to_date,
                                               from_time=from_date).limit(5)

    print detected_candidates

    for c in detected_candidates:
        df = pd.DataFrame(c.data)

        # Process the data. Select highest SNR in time (within the beam)
        df = df.sort_values('snr', ascending=False).drop_duplicates(subset=['time_sample', 'beam_id']).sort_values(
            by=['time_sample'])

        # Process the data. Select highest SNR in channel (within the beam)
        df = df.sort_values('snr', ascending=False).drop_duplicates(subset=['channel_sample', 'beam_id']).sort_values(
            by=['time_sample'])

        c.data = {
            'channel': df['channel'].tolist(),
            'time': df['time'].tolist(),
            'snr': df['snr'].tolist()
        }

    return detected_candidates.to_json()


@api_page.route('/api/status/birales_service', methods=['GET'])
def birales_status():
    # check the birales service status
    stat = os.system('service sshd status')

    cmd = '/bin/systemctl status %s.service' % 'birales'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stdout_list = proc.communicate()[0].split('\n')
    status = 'OFF'
    for line in stdout_list:
        if 'Active:' in line:
            if '(running)' in line:
                status = 'ON'
                break

    return json.dumps({
        'status': status,
        'msg': stdout_list,
    })


@api_page.route('/api/events', methods=['POST'])
def birales_events(from_date=None, to_date=None):
    if request.values.get('from_date'):
        from_date = dateutil.parser.parse(request.values.get('from_date'))

    if request.values.get('to_date'):
        to_date = dateutil.parser.parse(request.values.get('to_date'))

    events = Event.get(from_time=from_date, to_time=to_date)

    return json.dumps(
        {'events': events.to_json(), 'timestamp': datetime.datetime.utcnow().replace(tzinfo=pytz.utc).isoformat('T')})


@api_page.route('/api/stop', methods=['POST'])
def birales_pipeline_stop():
    try:
        broker.publish('birales_pipeline_control', 'KILL')
    except Exception:
        return json.dumps({
            'msg': 'An error has occurred when sending KILL command',
        }), 500
    else:
        return json.dumps({
            'msg': 'KILL command sent successfully',
        }), 200
