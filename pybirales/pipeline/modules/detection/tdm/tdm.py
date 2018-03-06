from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime
import logging as log
from pybirales import settings


class TDMWriter:
    def __init__(self):
        self._template_dir = os.path.dirname(__file__)
        self._template_filepath = 'input_template.tdm'
        self._tdm_mask = 'BIRALES_OUT_{:%Y%m%d}_{:0>3}.tdm'

        self._created_date = datetime.utcnow()
        self._created_date_str = '{:%Y%m%d}'.format(self._created_date)
        self._out_dir = self._create_out_dir(
            os.path.join(os.environ['HOME'], '.birales/tdm/out', self._created_date_str))

        self._template = Environment(loader=FileSystemLoader(self._template_dir)).get_template(self._template_filepath)

    @staticmethod
    def _create_out_dir(tdm_directory):
        if not os.path.exists(tdm_directory):
            os.makedirs(tdm_directory)

        return tdm_directory

    def _get_filename(self):
        return len([name for name in os.listdir(self._out_dir) if os.path.isfile(os.path.join(self._out_dir, name))]) + 1

    def write(self, filepath, obs_info, sd_track):
        data = dict(
            filename=obs_info['settings']['observation']['name'],
            creation_date=self._created_date.isoformat('T'),
            obs_info=obs_info,
            beams=sd_track.data['beam_id'].unique(),
            detection=sd_track.data,
            target_name=settings.target.name,
            tx=obs_info['transmitter_frequency'],
            integration_interval=obs_info['sampling_time']
        )

        # Parse the Jinja template using the provided data
        parsed_template = self._template.render(**data)

        # Default filepath for this space debris detection
        tdm_out_filepath = os.path.join(self._out_dir,
                                        self._tdm_mask.format(sd_track.data['time'].min(), self._get_filename()))

        # If filepath is provided, we overwrite the data in that file
        if filepath:
            tdm_out_filepath = os.path.join(self._out_dir, filepath)

        try:
            with open(tdm_out_filepath, "wb") as fh:
                fh.write(parsed_template)
                log.info('TDM file persisted at: {}'.format(tdm_out_filepath))
        except IOError:
            log.exception('TDM file could not be persisted at: {}'.format(tdm_out_filepath))
        finally:
            return tdm_out_filepath


class TDMReader:
    def __init__(self):
        pass


class CSVWriter:
    def __init__(self):
        self._template_dir = os.path.dirname(__file__)
        self._tdm_mask = 'BIRALES_OUT_{}_{:0>3}.csv'

        self._created_date = '{:%Y%m%d}'.format(datetime.utcnow())

        self._out_dir = self._create_out_dir(
            os.path.join(os.environ['HOME'], '.birales/debug/detection', self._created_date))

    @staticmethod
    def _create_out_dir(tdm_directory):
        if not os.path.exists(tdm_directory):
            os.makedirs(tdm_directory)

        return tdm_directory

    def _get_filename(self):
        return len([name for name in os.listdir(self._out_dir) if os.path.isfile(name)]) + 1

    def write(self, filepath, sd_track):
        # Default filepath for this space debris detection
        tdm_out_filepath = os.path.join(self._out_dir, self._tdm_mask.format(self._created_date, self._get_filename()))

        # If filepath is provided, we overwrite the data in that file
        if filepath:
            tdm_out_filepath = os.path.join(self._out_dir, filepath)

        try:
            sd_track.data.to_csv(tdm_out_filepath)
            log.debug('Space debris debug csv persisted at: {}'.format(tdm_out_filepath))
        except IOError:
            log.exception('Space debris debug csv could not be persisted at: {}'.format(tdm_out_filepath))
        finally:
            return tdm_out_filepath
