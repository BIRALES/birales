from flask import Blueprint

api = Blueprint('api', __name__, template_folder='templates')


@api.route('/observations', method=['POST'])
def create_obs():
    pass


@api.route('/observations/<observation_id>', method=['POST'])
def edit_obs(observation_id):
    pass


@api.route('/observations/<observation_id>', method=['POST'])
def delete_obs(observation_id):
    pass
