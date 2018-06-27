from flask import Blueprint
from flask import render_template, request
from flask_paginate import Pagination, get_page_parameter

from pybirales.repository.models import Event

events_page = Blueprint('events_page', __name__, template_folder='templates')

@events_page.route('/events')
def index():
    page = request.args.get(get_page_parameter(), default=1)
    per_page = 25

    events = Event.objects.order_by('-created_at').skip((page-1) * per_page).limit(per_page)
    pagination = Pagination(page=page, total=events.count(),
                            inner_window=5,
                            bs_version=3,
                            per_page=per_page,
                            record_name='observations')

    return render_template('modules/events/index.html',
                           events=events,
                           pagination=pagination)
