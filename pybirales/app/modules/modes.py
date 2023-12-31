import datetime
import os
import configparser
import logging as log
from flask import Blueprint, render_template, Response, json, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from pybirales.birales_config import BiralesConfig
from pybirales.app.modules.forms import ConfigurationForm
from pybirales.repository.models import Configuration as ConfigurationModel

import re
import ast
from flask_paginate import Pagination, get_page_parameter

configurations_page = Blueprint('configurations_page', __name__, template_folder='templates')

SETTINGS_FILEPATH = os.path.join(os.environ['HOME'], '.birales', 'configuration')
AVAILABLE_CONFIGS = {
    'dev': {
        'detection': {'filepath': 'templates/dev/detection.ini',
                      'description': 'Detection observations configuration template [DEV]'},
        'correlation': {'filepath': 'templates/dev/correlation.ini',
                        'description': 'Correlation configuration template [DEV]'},
    },
    'prod': {
        'detection': ('templates/prod/detection.ini', 'Detection configuration template [PROD]'),
        'calibration': ('templates/prod/calibration.ini', 'Calibration configuration template [PROD]'),
    },
    'uploaded': {

    }
}

ALLOWED_EXTENSIONS = ['ini']
CONFIG_MAP = {
    'dev': os.path.join(SETTINGS_FILEPATH, 'templates', 'dev'),
    'prod': os.path.join(SETTINGS_FILEPATH, 'templates', 'prod'),
    'uploads': os.path.join(SETTINGS_FILEPATH, 'uploads')
}


@configurations_page.route('/mode/<mode_id>')
def observation_mode(mode_id):
    """

    :return:
    """

    modes = {
        'detection': _read_config_files('templates/prod/detection.ini'),
        'calibration': _read_config_files('templates/prod/calibration.ini')
    }

    options = {
        'data': [],
        "draw": 1,
        "recordsTotal": 0,
        "recordsFiltered": 0,
    }

    for section in modes[mode_id].keys():
        for option in modes[mode_id][section]:
            options['data'].append([
                option, modes[mode_id][section][option], section
            ])
            options['recordsTotal'] += 1
            options['recordsFiltered'] += 1

    if mode_id in modes.keys():
        return Response(json.dumps(options), mimetype='application/json; charset=utf-8')


def _allowed_file(filename):
    """

    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_config_files(configuration):
    """

    :param configuration:
    :return:
    """
    config_parser = configparser.RawConfigParser()

    config_parser = _read_config_file(config_parser, 'birales.ini')
    config_parser = _read_config_file(config_parser, configuration)

    return {section: config_parser._sections[section] for section in config_parser._sections}


def _read_config_file(config_parser, configuration):
    try:
        with open(os.path.expanduser(os.path.join(SETTINGS_FILEPATH, configuration))) as f:
            config_parser.read_file(f)
        log.info('Config file at {} read successfully'.format(configuration))
    except IOError:
        log.info('Config file at {} was not found'.format(configuration))
    finally:
        return config_parser


def _config_to_dict(configuration_filepath):
    config_parser = configparser.RawConfigParser()

    config_parser = _read_config_file(config_parser, configuration='birales.ini')
    config_parser = _read_config_file(config_parser, configuration=configuration_filepath)

    sections_dict = {}
    # get sections and iterate over each
    sections = config_parser.sections()

    for section in sections:
        options = config_parser.options(section)
        temp_dict = {}
        for option in options:
            v = config_parser.get(section, option)

            temp_dict[option] = v
            # If value is a string, interpret it
            if isinstance(v, basestring):
                # Check if value is a number of boolean
                if re.match(re.compile("^True|False|[0-9]+(\.[0-9]*)?$"), v) is not None:
                    temp_dict[option] = ast.literal_eval(v)

                # Check if value is a list
                elif re.match("^\[.*\]$", re.sub('\s+', '', v)):
                    temp_dict[option] = ast.literal_eval(v)

        sections_dict[section] = temp_dict

    return sections_dict


def _get_config_files(root_path):
    _configs = []
    for directory, _, configs in os.walk(root_path):
        for uploaded_config in configs:
            _filepath = os.path.join(directory, uploaded_config)
            _configs.append({
                'name': os.path.splitext(uploaded_config)[0],
                'desc': 'Last modified on {:%Y-%m-%d}'.format(
                    datetime.datetime.fromtimestamp(os.path.getmtime(_filepath))),
                'env': os.path.basename(root_path),
                'file_name': _filepath
            })

    return _configs


def get_available_configs():
    configurations = []
    configurations.extend(_get_config_files(CONFIG_MAP['uploads']))
    configurations.extend(_get_config_files(CONFIG_MAP['dev']))
    configurations.extend(_get_config_files(CONFIG_MAP['prod']))

    return configurations


@configurations_page.route('/configurations')
def index():
    # Load the configuration files
    configurations = get_available_configs()

    page = request.args.get(get_page_parameter(), type=int, default=0)

    pagination = Pagination(page=page, total=len(configurations), bs_version=3, per_page=10,
                            record_name='configurations')

    return render_template('modules/configuration/configurations.html', configurations=configurations,
                           pagination=pagination)


@configurations_page.route('/configurations/upload', methods=['GET', 'POST'])
def upload_file():
    def upload_error(message):
        flash(message)
        log.exception(message)

        return Response(json.dumps(message), mimetype='application/json; charset=utf-8'), 500

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return upload_error('No file part')
        _file = request.files['file']
        if _file.filename == '':
            return upload_error('No selected file')
        if _file and _allowed_file(_file.filename):
            filename = secure_filename(_file.filename)
            filepath = os.path.join(CONFIG_MAP['uploads'], filename)
            if os.path.isfile(filepath):
                return upload_error('Filename is not unique. Please choose another name')

            flash('Configuration file uploaded')
            try:
                _file.save(filepath)
            except IOError:
                return upload_error('Could not save configuration')
            Response(json.dumps("Configuration Saved!"), mimetype='application/json; charset=utf-8'), 200

    return render_template('modules/configuration/upload.html')


@configurations_page.route('/configuration/<environment>/<configuration_name>')
def view(environment, configuration_name):
    try:
        env_configs = _get_config_files(CONFIG_MAP[environment])
        for env_config in env_configs:
            print(env_config)
            if env_config['name'] == configuration_name:
                return render_template('modules/configuration/configuration.html',
                                       configuration=_config_to_dict(env_config['file_name']),
                                       name=configuration_name)

        flash("Configurations file {} was not found".format(configuration_name), 'error')

        return redirect(url_for("configurations_page.index"))
    except KeyError:
        log.error('Configuration was not found')
        return redirect(url_for("configurations_page.index"))


@configurations_page.route('/configurations/edit', methods=['GET', 'POST'])
def edit():
    form = ConfigurationForm(request.form)
    if request.method == 'GET':
        configuration = ConfigurationModel.objects.order_by('-id').first()
        form = ConfigurationForm(obj=configuration)

    if request.method  == 'POST' and form.validate():
            try:
                _id = form.data['id']
                c_config_filepath = form.data['calibration_config_filepath']
                d_config_filepath = form.data['detection_config_filepath']
            except KeyError:
                flash('Validation error. Observation not submitted', 'error')
            else:
                if form.data['id'] == '':
                    configuration = ConfigurationModel(calibration_config_filepath=c_config_filepath,
                                                       detection_config_filepath=d_config_filepath)
                else:
                    configuration = ConfigurationModel.objects.get(pk=_id)
                    configuration.calibration_config_filepath = c_config_filepath
                    configuration.detection_config_filepath = d_config_filepath

                configuration.save()

                flash('Configuration updated successfully')

    return render_template('modules/configuration/index.html', form=form)
