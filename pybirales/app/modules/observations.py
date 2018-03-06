import mongoengine
from flask import Blueprint, render_template, request, abort, redirect, url_for
from flask_paginate import Pagination, get_page_parameter
from webargs import fields

from pybirales.repository.models import BeamCandidate
from pybirales.repository.models import Observation
from pybirales.app.modules.configurations import get_available_configs

observations_page = Blueprint('observations_page', __name__, template_folder='templates')

observations_args = {
    'from_time': fields.DateTime(missing=None, required=False),
    'to_time': fields.DateTime(missing=None, required=False),
}


@observations_page.route('/api/observation/<observation_id>/data')
def beam_data(observation_id):
    return BeamCandidate.get(observation_id=observation_id).to_json()


@observations_page.route('/observations')
def index():
    """
    Serve the client-side application

    :return:
    """

    page = request.args.get(get_page_parameter(), type=int, default=0)
    per_page = 10
    observations = Observation.objects.order_by('-date_time_start').skip(page * per_page).limit(per_page)
    pagination = Pagination(page=page, total=observations.count(),
                            inner_window=5,
                            bs_version=3,
                            per_page=per_page,
                            record_name='observations')

    return render_template('modules/observations.html',
                           observations=observations,
                           pagination=pagination)


@observations_page.route('/observations/create')
def create():
    configurations = get_available_configs()
    return render_template('modules/observation_create.html', configurations=configurations)


@observations_page.route('/observations/<observation_id>')
def view(observation_id):
    try:
        observation = Observation.objects.get(id=observation_id)
        return render_template('modules/observation.html', observation=observation)
    except mongoengine.DoesNotExist:
        abort(404)


@observations_page.route('/observations/edit/<observation_id>')
def edit(observation_id):
    return observation_id


@observations_page.route('/observations/delete/<observation_id>')
def delete(observation_id):
    try:
        observation = Observation.objects.get(id=observation_id)
        observation.delete()
        return redirect(url_for('observations_page.index'))
    except mongoengine.DoesNotExist:
        abort(404)
