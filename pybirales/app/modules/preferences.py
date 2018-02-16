import os
from configparser import ConfigParser
from flask import Blueprint

preferences_page = Blueprint('preferences_page', __name__, template_folder='templates')

SETTINGS_FILEPATH = os.path.join(os.environ['HOME'])


@preferences_page.route('/preferences')
def index():

    pass


@preferences_page.route('/preferences/create')
def create():
    pass


@preferences_page.route('/preferences/<filename>')
def view(filename):
    pass



