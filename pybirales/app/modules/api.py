from flask import Blueprint
from pybirales.repository.models import SpaceDebrisTrack

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