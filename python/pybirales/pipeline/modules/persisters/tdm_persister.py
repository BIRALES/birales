import logging as log
import os
from datetime import datetime

import numpy as np
from jinja2 import Environment, FileSystemLoader
from scipy import io

from pybirales import settings
from pybirales.events.events import TDMCreatedEvent
from pybirales.events.publisher import publish
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


class TDMPersister(ProcessingModule):
    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(TDMPersister, self).__init__(config, input_blob)

        self._template_dir = os.path.dirname(__file__)
        self._created_date = datetime.utcnow()
        self._created_date_str = '{:%Y%m%d}'.format(self._created_date)

        self._filename_mask = settings.instrument.name_prefix + '_BIRALES_OUT_{:%Y%m%dT%H%M%S}.tdm'
        self._start_time = '{}_{:%H%M%S}'.format(settings.observation.name, self._created_date)

        self._out_dir = os.path.join(os.environ['HOME'], '.birales/tdm/out', self._created_date_str, self._start_time)
        self._template_filepath = 'input_template.tdm'
        self._template = Environment(loader=FileSystemLoader(self._template_dir)).get_template(self._template_filepath)

        self._create_out_dir()

        self._detection_num = 1

        # Processing module name
        self.name = "TDM Persister"

    def _create_out_dir(self):
        """

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

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._input.shape, datatype=float)

    def _write(self, obs_info, obs_name, target_name, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """

        filepath = self._get_filename(sd_track, detection_num)
        beam_noise = list(np.mean(obs_info['channel_noise'], axis=1))
        data = dict(
            filename=obs_name,
            beam_noise=beam_noise,
            creation_date=datetime.utcnow().isoformat('T'),
            beams=np.unique(sd_track.data['beam_id']),
            detection=sd_track.reduce_data(remove_duplicate_epoch=True, remove_duplicate_channel=True),
            target_name=target_name,
            tx=obs_info['transmitter_frequency'],
            pointings={
                'ra_dec': obs_info['pointings'],
                'az_el': obs_info['beam_az_el'].tolist(),
                'declination': obs_info['declination']
            },
            integration_interval=obs_info['sampling_time']
        )

        # Parse the Jinja template using the provided data
        parsed_template: str = self._template.render(**data)

        try:
            with open(filepath, "wb") as fh:
                # noinspection PyTypeChecker
                fh.write(parsed_template)
                log.info('Outputted TDM {} persisted at: {} for track {}'.format(detection_num, filepath, sd_track.id))

                self._detection_num += 1
        except IOError:
            log.exception(
                'Output TDM {} could not be persisted at: {} for track {}'.format(detection_num, filepath, sd_track.id))
        else:
            publish(TDMCreatedEvent(sd_track, filepath))

        if settings.detection.debug_candidates:
            np.save(filepath + '.npy', sd_track)

            self.save_mat(filepath + '.mat', obs_name, target_name, obs_info, sd_track)

    def save_mat(self, filename, obs_name, target_name, obs_info, sd_track):
        io.savemat(filename, dict(
            filename=obs_name,
            beam_noise=list(np.mean(obs_info['channel_noise'], axis=1)),
            creation_date=datetime.utcnow().isoformat('T'),
            beams=np.unique(sd_track.data['beam_id']),
            detection=list(sd_track.reduce_data(remove_duplicate_epoch=True, remove_duplicate_channel=True)),
            target_name=target_name,
            tx=obs_info['transmitter_frequency'],
            pointings={
                'ra_dec': obs_info['pointings'],
                'az_el': obs_info['beam_az_el'].tolist(),
                'declination': obs_info['declination']
            },
            integration_interval=obs_info['sampling_time']
        ), do_compression=True)

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        if self._iteration_counter < 1:
            return


        tracks_to_output = obs_info['transitted_tracks']
        tracks_to_output = tracks_to_output.extend(obs_info['transitted_tracks_msds'])

        obs_name = settings.observation.name
        target_name = settings.observation.target_name

        # print tracks_to_output, obs_info['transitted_tracks'], obs_info['transitted_tracks_msds']
        if tracks_to_output:
            for sd_track in tracks_to_output:
                self._write(obs_info, obs_name, target_name, sd_track, self._detection_num)

        return obs_info
