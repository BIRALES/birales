import datetime
import logging
import logging as log
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


class TLE_Target:
    def __init__(self, name, transit_time, doppler):
        self.name = name
        self.doppler = doppler
        self.transit_time = datetime.datetime.strptime(transit_time, '%d %b %Y %H:%M:%S.%f')


class FitsPersister(ProcessingModule):
    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(FitsPersister, self).__init__(config, input_blob)

        # Counter
        self._counter = 0

        self._filename_suffix = 'raw'

        # Processing module name
        self.name = "FitsPersister"

        # Get the destination file path of the persisted data
        self._fits_filepath = self._get_filepath()

        self._fits_file = None
        self._header = None

        self._beams_to_visualise = None

        # A new fits file will be created every Chunk Size iterations
        self._chuck_size = 10

        self.tle_targets = [
            TLE_Target(name='norad_1328', transit_time='05 MAR 2019 10:37:53.01', doppler=-1462.13774),
            # TLE_Target(name='norad_41182', transit_time='05 MAR 2019 11:23:28.73', doppler=-3226.36329)
        ]

    def _get_filepath(self, counter=0):
        """
        Get the file path for the fits file

        :return:
        """
        directory = os.path.join(os.environ['HOME'],
                                 settings.fits_persister.visualise_fits_dir,
                                 settings.observation.name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return os.path.join(directory,
                            '{}_{}_{}.fits'.format(settings.observation.name, self._filename_suffix, counter))

    def _tear_down(self):
        """
        Tear down function for the persister module

        :return:
        """

        # Close the opened fits file handler
        if self._fits_file:
            self._fits_file.close()

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._input.shape, datatype=np.float64)

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        if self._iter_count < 2:
            return

        # print obs_info['start_center_frequency'], obs_info['start_center_frequency'] + obs_info['channel_bandwidth'] * 8192

        self._fits_filepath = self._get_filepath(int(np.floor(self._iter_count / self._chuck_size)))

        # Append data to the body of the fits file
        if self._beams_to_visualise:
            # new_data = input_data[[15], :, :]

            new_data = input_data[self._beams_to_visualise, :, :]
            new_data = np.power(np.abs(new_data), 2.0)

            for target in self.tle_targets:
                self.plot_TLE(obs_info, new_data[0, ...], target)

            try:
                self._fits_file = fits.open(self._fits_filepath)
                new_data = np.dstack([self._fits_file[0].data, new_data])
                self._write_to_file(new_data)
                logging.info(f"Saved beam ({self._beams_to_visualise}) data to {self._fits_filepath}")
            except IOError:
                # Fits file not created, create a new one (and set header)
                self._header = self._create_header(obs_info, settings.observation.name)
                self._write_to_file(new_data, self._header)

        output_data[:] = input_data[:]

    def _write_to_file(self, data, header=None):
        """
        Create or append the input data to the fits file

        :param data:
        :param header:
        :return:
        """
        try:
            if header:
                fits.writeto(self._fits_filepath, data, overwrite=True, header=header)
                log.debug('Created new fits file at {}'.format(self._fits_filepath))
            else:
                log.debug('Appending new data to the fits file at {}'.format(self._fits_filepath))
                fits.writeto(self._fits_filepath, data, overwrite=True)
        except KeyError as e:
            log.exception("Could not save the data. Input data is not valid.")

    @staticmethod
    def _create_header(obs_info, observation_name):
        """
        Create the header for the fits file

        :param obs_info:
        :param observation_name:
        :return:
        """
        header = fits.Header()
        header.set('OBS', observation_name)
        header.set('S_FREQ', obs_info['start_center_frequency'])
        header.set('CHL_BAND', obs_info['channel_bandwidth'])
        header.set('DATE', obs_info['timestamp'].isoformat())
        header.set('NSAMP', obs_info['nsamp'])
        header.set('TX', obs_info['transmitter_frequency'])

        return header

    def plot_TLE(self, obs_info, input_data, tle_target):
        expected_transit_time = tle_target.transit_time
        target_name = tle_target.name
        expected_doppler = tle_target.doppler

        start_window = expected_transit_time - datetime.timedelta(seconds=30)
        end_window = expected_transit_time + datetime.timedelta(seconds=20)

        print('Current time is', obs_info['timestamp'])
        print('TLE is between {} and {}'.format(start_window, end_window))
        print('Persister Within window', end_window >= obs_info['timestamp'] >= start_window, self._iter_count)

        if end_window >= obs_info['timestamp'] >= start_window:
            expected_channel = (obs_info['transmitter_frequency'] * 1e6 + expected_doppler) - obs_info[
                'start_center_frequency'] * 1e6
            expected_channel /= obs_info['channel_bandwidth'] * 1e6

            expected_channel = int(expected_channel)

            channel_window = expected_channel - 500, expected_channel + 500

            filename = '{}_TLE_prediction_{}'.format(target_name, self._iter_count)

            time = [obs_info['timestamp'], datetime.timedelta(seconds=obs_info['sampling_time'] * 160) + obs_info[
                'timestamp']]

            subset = input_data[channel_window[0]: channel_window[1], :]

            print('Expecting target to be between', channel_window, subset.shape)
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Channel")
            im = ax.imshow(subset, aspect='auto', interpolation='none', origin='lower', vmin=0,
                           extent=[0, 160, channel_window[0], channel_window[1]])
            fig.colorbar(im)
            fig.tight_layout()
            plt.title(filename + ' from {:%H.%M.%S} to {:%M.%S}'.format(time[0], time[1]))
            plt.show()

            fig.savefig(filename, bbox_inches='tight')
            fig.clf()


class RawDataFitsPersister(FitsPersister):
    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(RawDataFitsPersister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'visualise_raw_beams', 'visualise_fits_dir'} - set(config.settings()) != set():
            raise PipelineError(
                "Persister: Missing keys on configuration. ('visualise_raw_beams', 'visualise_fits_dir')")

        self._beams_to_visualise = settings.fits_persister.visualise_raw_beams

        self._filename_suffix = 'raw'

        # Processing module name
        self.name = "RawFitsPersister"

        # Get the destination file path of the persisted data
        self._fits_filepath = self._get_filepath()


class FilteredDataFitsPersister(FitsPersister):
    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(FilteredDataFitsPersister, self).__init__(config, input_blob)
        # Sanity checks on configuration
        if {'visualise_filtered_beams', 'visualise_fits_dir'} - set(config.settings()) != set():
            raise PipelineError(
                "Persister: Missing keys on configuration. ('visualise_raw_beams', 'visualise_fits_dir')")

        self._beams_to_visualise = settings.fits_persister.visualise_filtered_beams

        self._filename_suffix = 'filtered'

        # Processing module name
        self.name = "FilteredFitsPersister"

        # Get the destination file path of the persisted data
        self._fits_filepath = self._get_filepath()
