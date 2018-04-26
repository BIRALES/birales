import logging as log

import mongoengine
from flask import Blueprint
from flask import render_template, request, abort, redirect, url_for
from flask_paginate import Pagination, get_page_parameter

from pybirales.app.modules.forms import DetectionModeForm
from pybirales.repository.models import Observation
from pybirales.repository.models import SpaceDebrisTrack

observations_page = Blueprint('observations_page', __name__, template_folder='templates')

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

    return render_template('modules/observations/index.html',
                           observations=observations,
                           pagination=pagination)


@observations_page.route('/observations/<mode>/create', methods=['POST', 'GET'])
def create(mode):
    if mode == 'detection':
        form = DetectionModeForm(request.form)

        if request.method == 'POST' and form.validate():
            # Observation is valid and can be submitted to service
            print form.data
            pass
        return render_template('modules/observations/create/detection.html', form=form)
    else:
        log.error('Observation mode is not valid')
        abort(422)




@observations_page.route('/observations/<observation_id>')
def view(observation_id):
    try:
        observation = Observation.objects.get(id=observation_id)

        tracks = SpaceDebrisTrack.get(observation_id=observation_id)
        return render_template('modules/observations/view.html', observation=observation, tracks=tracks)
    except mongoengine.DoesNotExist:
        log.exception('Database error')
        abort(503)

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
