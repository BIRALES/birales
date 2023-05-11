import csv
import logging as log
import os
from datetime import datetime

import numpy as np
import socket
from jinja2 import Environment, FileSystemLoader
from scipy import io

from pybirales import settings
from pybirales.events.events import TDMCreatedEvent
from pybirales.events.publisher import publish
from pybirales.utilities.singleton import Singleton


def persist(obs_info, track):
    """

    :param obs_info:
    :param obs_name:
    :param target_name:
    :param track:
    :param debug:
    :return:
    """

    _tdm_persister.persist_track(obs_info, track)


@Singleton
class TDMPersister:
    def __init__(self):

        self._template_dir = os.path.dirname(__file__)

        name_prefix = settings.instrument.name_prefix

        if socket.gethostname() == 'med-frb':
            name_prefix = '2N'

        self._filename_mask = name_prefix + '_BIRALES_OUT_{:%Y%m%dT%H%M%S}.tdm'

        self._template_filepath = 'input_template.tdm'
        self._template = Environment(loader=FileSystemLoader(self._template_dir)).get_template(self._template_filepath)

        self._detection_num = 1

    @staticmethod
    def _create_out_dir(out_dir):
        """

        :param out_dir:
        :return:
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        return out_dir

    def _get_filename(self, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """
        utc_now = datetime.utcnow()
        created_date = '{:%Y%m%d}'.format(utc_now)
        out_dir = os.path.join(os.environ['HOME'], '.birales/tdm/out', created_date, settings.observation.name)

        out_dir = self._create_out_dir(out_dir)

        return os.path.join(out_dir, self._filename_mask.format(min(sd_track.data['time'])))

    def _get_debug_filename(self, obs_name, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """
        utc_now = datetime.utcnow()
        created_date = '{:%Y%m%d}'.format(utc_now)
        out_dir = os.path.join(os.environ['HOME'], '.birales/debug/detection/', created_date, obs_name)

        out_dir = self._create_out_dir(out_dir)

        return os.path.join(out_dir, 'RSO_{}'.format(detection_num))

    def _write(self, obs_info, sd_track, detection_num):
        """

        :param sd_track:
        :param detection_num:
        :return:
        """

        filepath = self._get_filename(sd_track, detection_num)
        beam_noise = list(np.mean(obs_info['channel_noise'], axis=1))
        data = dict(
            filename=settings.observation.name,
            beam_noise=beam_noise,
            creation_date=datetime.utcnow().isoformat('T'),
            beams=np.unique(sd_track.data['beam_id']),
            detection=sd_track.reduce_data(remove_duplicate_epoch=True, remove_duplicate_channel=True),
            target_name=settings.observation.target_name,
            tx=obs_info['transmitter_frequency'],
            pointings={
                'ra_dec': obs_info['pointings'],
                'az_el': obs_info['beam_az_el'].tolist(),
                'declination': obs_info['declination']
            },
            integration_interval=obs_info['sampling_time']
        )

        # Parse the Jinja template using the provided data
        parsed_template = self._template.render(**data)

        try:
            with open(filepath, "wb") as fh:
                fh.write(parsed_template.encode('utf-8'))
                log.info('Outputted TDM {} persisted at: {} for track {}'.format(detection_num, filepath, sd_track.id))
        except IOError:
            log.exception(
                'Output TDM {} could not be persisted at: {} for track {}'.format(detection_num, filepath, sd_track.id))
        else:
            publish(TDMCreatedEvent(sd_track, filepath))

    def save_mat(self, obs_info, sd_track, detection_num):
        filepath = self._get_debug_filename(settings.observation.name, detection_num)

        reduced = sd_track.reduce_data()
        io.savemat(filepath, dict(
            filename=settings.observation.name,
            raw_filepath=settings.rawdatareader.filepath,
            beam_noise=list(np.mean(obs_info['channel_noise'], axis=1)),
            creation_date=datetime.utcnow().isoformat('T'),
            beams=np.unique(sd_track.data['beam_id']),
            detection=list(sd_track.reduce_data(remove_duplicate_epoch=True, remove_duplicate_channel=True)),
            data={
                'time': [str(t) for t in sd_track.data['time']],
                'time_sample': sd_track.data['time_sample'].tolist(),
                'channel': sd_track.data['channel'].tolist(),
                'channel_sample': sd_track.data['channel_sample'].tolist(),
                'snr': sd_track.data['snr'].tolist(),
                'beam_id': sd_track.data['beam_id'].tolist(),
            },
            target_name=settings.observation.target_name,
            tx=obs_info['transmitter_frequency'],
            pointings={
                'ra_dec': obs_info['pointings'],
                'az_el': obs_info['beam_az_el'],
                'declination': obs_info['declination']
            },
            reference=sd_track.ref_data,
            sampling_time=obs_info['sampling_time'],
            reduced_data={
                'time': [str(t) for t in reduced['time']],
                'time_sample': reduced['time_sample'].tolist(),
                'channel': reduced['channel'].tolist(),
                'channel_sample': reduced['channel_sample'].tolist(),
                'snr': reduced['snr'].tolist(),
                'beam_id': reduced['beam_id'].tolist(),
            }
        ), do_compression=True)

        # Save a summary of the detection data to csv
        summary_filepath = os.path.join(os.path.dirname(filepath), '../summary.csv')

        mid = sd_track.data['snr'].idxmax()

        doppler = (sd_track.data['channel'] - obs_info['transmitter_frequency']) * 1e6
        row = {
            'name': settings.observation.name,
            'psnr_doppler': doppler.iloc[mid],
            'psnr_timestamp': sd_track.data.iloc[mid]['time'].isoformat('T'),
            'psnr': sd_track.data['snr'].max(),
            'mean_snr': np.mean(sd_track.data['snr']),
            'activated_beams': ','.join([str(b) for b in sd_track.data['beam_id'].unique().tolist()]),
            'n': len(sd_track.data['channel']),
            'r_value': sd_track.r_value,
            'filepath': filepath,
            'created_at': sd_track._created_at,
            'ref_doppler': sd_track.ref_data['doppler'],
            'ref_time': sd_track.ref_data['time'].isoformat('T'),
            'ref_snr': sd_track.ref_data['snr'],
            'iterations': ','.join([str(b) for b in sd_track.data['iter'].unique().tolist()]),
        }

        with open(summary_filepath, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(row)

    def persist_track(self, obs_info, track):
        # Save TDM file

        if settings.detection.save_tdm:
            self._write(obs_info, track, self._detection_num)

        if settings.detection.debug_candidates:
            # Save MAT file
            self.save_mat(obs_info, track, self._detection_num)

        self._detection_num += 1


_tdm_persister = TDMPersister.Instance()
