from flask import Blueprint, render_template, request
from pybirales.repository.repository import ObservationsRepository
from webargs import fields
from webargs.flaskparser import use_args
from astropy.time import Time, TimeDelta
from flask_paginate import Pagination, get_page_parameter

observations_page = Blueprint('observations_page', __name__, template_folder='templates')
observations_repo = ObservationsRepository()

observations_args = {
    'from_time': fields.DateTime(missing=None, required=False),
    'to_time': fields.DateTime(missing=None, required=False),
}


@observations_page.route('/observations')
def index():
    """
    Serve the client-side application

    :param args:
    :return:
    """

    page = request.args.get(get_page_parameter(), type=int, default=1)

    observations = observations_repo.database.configurations.find()
    pagination = Pagination(page=page, total=10,
                            inner_window=5,
                            bs_version=3,
                            per_page=5,
                            record_name='observations')

    return render_template('modules/observations.html',
                           observations=observations,
                           pagination=pagination)
