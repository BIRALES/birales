import datetime
import json
import logging as log

import mongoengine
from flask import Blueprint
from flask import render_template, request, abort, redirect, url_for
from flask_paginate import Pagination, get_page_parameter

from pybirales.app.modules.forms import DetectionModeForm
from pybirales.repository.message_broker import RedisManager
from pybirales.repository.message_broker import broker
from pybirales.repository.models import Observation
from pybirales.repository.models import SpaceDebrisTrack

observations_page = Blueprint('observations_page', __name__, template_folder='templates')
OBSERVATIONS_CHL = 'birales_scheduled_obs'
OBSERVATIONS_DEL_CHL = 'birales_delete_obs'

@observations_page.route('/observations')
def index():
    """
    Serve the client-side application

    :return:
    """

    page = request.args.get(get_page_parameter(), default=1)
    per_page = 10

    observations = Observation.objects.order_by('-date_time_start').skip((page-1) * per_page).limit(per_page)
    pagination = Pagination(page=page, total=observations.count(),
                            inner_window=5,
                            bs_version=3,
                            per_page=per_page,
                            record_name='observations')

    return render_template('modules/observations/index.html',
                           observations=observations,
                           pagination=pagination)


def _observation_from_form(form_data, mode):
    def json_convertor(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    duration = (form_data['date_end'] - form_data['date_start']).total_seconds()

    obs_name = form_data['obs_name']
    obs_config = "pybirales/configuration/templates/dev/detection.ini"
    obs_pipeline = "detection_pipeline"
    obs_type = 'observation'
    if mode == 'calibration':
        obs_config = "pybirales/configuration/templates/dev/calibration.ini"
        obs_pipeline = "correlation_pipeline"
        obs_type = 'calibration'

    return json.dumps({
        "name": obs_name,
        "type" : obs_type,
        "pipeline": obs_pipeline,
        "config_file": [
            "pybirales/configuration/birales.ini",
            obs_config
        ],
        "config_parameters": {
            "beamformer": {
                "reference_declination": float(form_data['declination'])
            },
            "observation": {
                "name": obs_name,
                "transmitter_frequency": float(form_data['transmitter_frequency']),
            },
            "target": {
                "name": form_data['target_name'],
            },
            "start_time": form_data['date_start'],
            "duration": duration
        }
    }, default=json_convertor)


@observations_page.route('/observations/<mode>/create', methods=['POST', 'GET'])
def create(mode):
    if mode == 'detection':
        form = DetectionModeForm(request.form)
        if request.method == 'POST' and form.validate():
            # Observation is valid and can be submitted to service
            obs_data = _observation_from_form(form.data, mode)

            # Publish the observation to the BIRALES scheduler
            broker.publish(OBSERVATIONS_CHL, obs_data)
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


@observations_page.route('/observations/<observation_id>/logs')
def observation_logs(observation_id):
    try:
        observation = Observation.objects.get(id=observation_id)
        return json.dumps({'log_files': observation.log_files})
    except mongoengine.DoesNotExist:
        log.exception('Database error')
        abort(503)


@observations_page.route('/observations/<observation_id>/logs/tail')
def observation_logs_tail(observation_id):
    try:
        observation = Observation.objects.get(id=observation_id)
    except mongoengine.DoesNotExist:
        log.exception('Database error')
        abort(503)
        return

    try:
        with open(observation.log_filepath) as f:
            data = f.readlines()
        tail = data[-10:]
        return render_template('modules/observations/tail.html', tail=tail)
    except IOError:
        log.error('Log file for observation %s does not exist', observation_id)


@observations_page.route('/observations/edit/<observation_id>')
def edit(observation_id):
    return observation_id


@observations_page.route('/observations/delete/<observation_id>')
def delete(observation_id):
    try:
        # Scheduler should reload itself if changes were made to the schedule
        broker.publish(OBSERVATIONS_DEL_CHL, json.dumps({
            'obs_id': observation_id
        }))

        return redirect(url_for('observations_page.index'))
    except mongoengine.DoesNotExist:
        abort(404)
