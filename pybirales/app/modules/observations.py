from flask import Blueprint

observations_page = Blueprint('observations_page', __name__, template_folder='templates')


@observations_page.route('/')
def index():
    pass
