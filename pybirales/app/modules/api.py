from flask import Blueprint, request
from pybirales.repository.models import SpaceDebrisTrack
import dateutil.parser

api_page = Blueprint('api_page', __name__, template_folder='templates')


@api_page.route('/observations', methods=['POST'])
def create_obs():
    pass


@api_page.route('/observations/<observation_id>', methods=['POST'])
def edit_obs(observation_id):
    pass


@api_page.route('/observations/<observation_id>', methods=['POST'])
def delete_obs(observation_id):
    pass


@api_page.route('/tracks/<track_id>', methods=['GET'])
def track(track_id):
    tracks = SpaceDebrisTrack.objects.get(pk=track_id)
    return tracks.to_json()


@api_page.route('/api/live/data', methods=['POST'])
def observation_track_data(observation_id=None, from_date=None, to_date=None):
    if request.values.get('from_date'):
        from_date = dateutil.parser.parse(request.values.get('from_date'))

    if request.values.get('to_date'):
        to_date = dateutil.parser.parse(request.values.get('to_date'))

    detected_candidates = SpaceDebrisTrack.get(observation_id=observation_id, to_time=to_date, from_time=from_date)

    return detected_candidates.to_json()
