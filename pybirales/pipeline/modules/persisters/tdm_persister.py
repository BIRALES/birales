import logging as log
import os
from datetime import datetime

import numpy as np
from jinja2 import Environment, FileSystemLoader

from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales import settings

class TDMPersister(ProcessingModule):
    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(TDMPersister, self).__init__(config, input_blob)

        self._template_dir = os.path.dirname(__file__)
        self._created_date = datetime.utcnow()
        self._created_date_str = '{:%Y%m%d}'.format(self._created_date)

        self._filename_mask = 'BIRALES_OUT_{:%Y%m%d}_{:0>3}.tdm'
        self._out_dir = os.path.join(os.environ['HOME'], '.birales/tdm/out', self._created_date_str)
        self._template_filepath = 'input_template.tdm'
        self._template = Environment(loader=FileSystemLoader(self._template_dir)).get_template(self._template_filepath)

        self._create_out_dir()

        self._detection_num = 1

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
        return os.path.join(self._out_dir, self._filename_mask.format(min(sd_track.data['time']), detection_num))

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._config, self._input.shape, datatype=np.float)

    def _write(self, obs_name, target_name, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """

        filepath = self._get_filename(sd_track, detection_num)
        data = dict(
            filename=obs_name,
            creation_date=self._created_date.isoformat('T'),
            beams=np.unique(sd_track.data['beam_id']),
            detection=sd_track.data,
            target_name=target_name,
            tx=sd_track.tx,
            pointings=sd_track.pointings,
            integration_interval=sd_track.sampling_time
        )

        # Parse the Jinja template using the provided data
        parsed_template = self._template.render(**data)

        try:
            with open(filepath, "wb") as fh:
                fh.write(parsed_template)
                log.info('Outputted TDM {} persisted at: {} for track {}'.format(detection_num, filepath, sd_track.id))

                self._detection_num += 1
        except IOError:
            log.exception(
                'Output TDM {} could not be persisted at: {} for track {}'.format(detection_num, filepath, sd_track.id))

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        if self._iter_count < 1:
            return

        tracks_to_output = obs_info['transitted_tracks']

        obs_name = settings.observation.name
        target_name = settings.observation.target_name

        for sd_track in tracks_to_output:
            self._write(obs_name, target_name, sd_track, self._detection_num)

        return obs_info