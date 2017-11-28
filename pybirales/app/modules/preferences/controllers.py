from flask import Blueprint

preferences_page = Blueprint('preferences_page', __name__, template_folder='templates')


@preferences_page.route('/')
def index():
    pass
