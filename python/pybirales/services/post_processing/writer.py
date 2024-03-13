import logging as log
import os
from abc import abstractmethod
from datetime import datetime

import numpy as np
from jinja2 import Environment, FileSystemLoader

from pybirales import settings


class Writer:
    """

    """

    def __init__(self):
        """

        :param observation:
        """
        self._template_dir = os.path.dirname(__file__)
        self._created_date = datetime.utcnow()
        self._created_date_str = '{:%Y%m%d}'.format(self._created_date)

        # To be overridden by sub-class
        self._filename_mask = None

        # To be overridden by sub-class
        self._out_dir = None

    def _create_out_dir(self):
        """

        :param out_dir:
        :return:
        """

        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

        return self._out_dir

    def _get_filename(self, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """
        return os.path.join(self._out_dir, self._filename_mask.format(min(sd_track.data['time'])))

    @abstractmethod
    def write(self, observation, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """
        pass


class DebugCandidatesWriter(Writer):
    """

    """

    def __init__(self):
        """

        :param observation:
        """
        Writer.__init__(self)

        self._filename_mask = 'BIRALES_OUT_{:%Y%m%dT%H%M%S}.csv'
        self._out_dir = os.path.join(os.environ['HOME'], '.birales/debug/detection', self._created_date_str)

        self._create_out_dir()

    def write(self, observation, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """
        filepath = self._get_filename(sd_track, detection_num)
        try:
            sd_track.data.to_csv(filepath)
            log.debug('Debug Candidate CSV file {} persisted at: {}'.format(detection_num, filepath))
        except IOError:
            log.exception('Debug Candidate CSV file {} could not be persisted at: {}'.format(detection_num,filepath))


class TDMWriter(Writer):
    """

    """

    def __init__(self):
        """

        :param observation:
        """
        Writer.__init__(self)

        self._filename_mask = 'BIRALES_OUT_{:%Y%m%dT%H%M%S}.tdm'
        self._out_dir = os.path.join(os.environ['HOME'], '.birales/tdm/out', self._created_date_str)
        self._template_filepath = 'input_template.tdm'
        self._template = Environment(loader=FileSystemLoader(self._template_dir)).get_template(self._template_filepath)

        self._create_out_dir()

    def write(self, observation, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """

        filepath = self._get_filename(sd_track, detection_num)
        data = dict(
            filename=observation.name,
            creation_date=self._created_date.isoformat('T'),
            beams=np.unique(sd_track.data['beam_id']),
            detection=sd_track.data,
            target_name=settings.observation.target_name,
            tx=sd_track.tx,
            pointings=sd_track.pointings,
            integration_interval=sd_track.sampling_time
        )

        # Parse the Jinja template using the provided data
        parsed_template = self._template.render(**data)

        try:
            with open(filepath, "wb") as fh:
                fh.write(parsed_template)
                log.info('Output TDM {} persisted at: {}'.format(detection_num, filepath))
        except IOError:
            log.exception('Output TDM {} could not be persisted at: {}'.format(detection_num, filepath))
