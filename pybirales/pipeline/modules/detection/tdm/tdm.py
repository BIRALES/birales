import logging as log
import os
from datetime import datetime
from threading import Thread, Event

from jinja2 import Environment, FileSystemLoader

from pybirales import settings


class TDMWriter(Thread):
    def __init__(self, queue):
        self.queue = queue

        super(TDMWriter, self).__init__()

        self.name = 'TDM Writer'
        self.daemon = True

        self._template_dir = os.path.dirname(__file__)
        self._template_filepath = 'input_template.tdm'
        self._tdm_mask = 'BIRALES_OUT_{:%Y%m%d}_{:0>3}.tdm'
        self._csv_mask = 'BIRALES_OUT_{:%Y%m%d}_{:0>3}.csv'

        self._created_date = datetime.utcnow()
        self._created_date_str = '{:%Y%m%d}'.format(self._created_date)
        self._tdm_out_dir = self._create_out_dir(
            os.path.join(os.environ['HOME'], '.birales/tdm/out', self._created_date_str))
        self._csv_out_dir = self._create_out_dir(
            os.path.join(os.environ['HOME'], '.birales/debug/detection', self._created_date_str))

        self._template = Environment(loader=FileSystemLoader(self._template_dir)).get_template(self._template_filepath)

        self._stop_event = Event()

    def run(self):
        while not self._stop_event.is_set():
            sd, obs_info = self.queue.get()

            self._write(obs_info, sd)

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def _create_out_dir(out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        return out_dir

    def _write(self, obs_info, sd_track):
        # Default filepath for this space debris detection
        tdm_out_filepath = os.path.join(self._tdm_out_dir,
                                        self._tdm_mask.format(sd_track.data['time'].min(), sd_track.detection_num))

        csv_out_filepath = os.path.join(self._csv_out_dir,
                                        self._csv_mask.format(sd_track.data['time'].min(), sd_track.detection_num))

        # Write the TDM file
        if settings.detection.save_tdm:
            self._write_tdm(tdm_out_filepath, obs_info, sd_track)

        # Write the CSV file
        if settings.detection.debug_candidates:
            self._write_csv(csv_out_filepath, sd_track)

    def _write_tdm(self, tdm_out_filepath, obs_info, sd_track):

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

        try:
            with open(tdm_out_filepath, "wb") as fh:
                fh.write(parsed_template)
                log.info('TDM file persisted at: {}'.format(tdm_out_filepath))
        except IOError:
            log.exception('TDM file could not be persisted at: {}'.format(tdm_out_filepath))

    def _write_csv(self, csv_out_filepath, sd_track):
        try:
            sd_track.data.to_csv(csv_out_filepath)
            log.debug('Space debris debug csv persisted at: {}'.format(csv_out_filepath))
        except IOError:
            log.exception('Space debris debug csv could not be persisted at: {}'.format(csv_out_filepath))


class TDMReader:
    def __init__(self):
        pass
