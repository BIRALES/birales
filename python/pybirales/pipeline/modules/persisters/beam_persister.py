import datetime
import os
import h5py

import numpy as np

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob

cu = None
try:
    import cupy as cu
except ImportError:
    pass


class BeamPersister(ProcessingModule):

    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(BeamPersister, self).__init__(config, input_blob)

        # Create directory if it doesn't exist
        dir_path = os.path.expanduser(settings.persisters.directory)
        directory = os.path.join(dir_path, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                                 settings.observation.name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        self._beam_file = os.path.join(directory, f"{settings.observation.name}_beams.h5")

        # Open file (if file exists, remove first)
        if os.path.exists(self._beam_file):
            os.remove(self._beam_file)

        # Iteration counter
        self._counter = 0

        # Processing module name
        self.name = "Persister"

    def _create_output_file(self, obs_info):
        """ Create the HDF5 output file, including metadata """
        with h5py.File(self._beam_file, 'w') as f:
            # Create group that will contain observation information
            info = f.create_group('observation_info')

            # Add attributes to group
            info.attrs['observation_name'] = settings.observation.name
            info.attrs['transmitter_frequency'] = obs_info['transmitter_frequency']
            info.attrs['sampling_time'] = obs_info['sampling_time']
            info.attrs['start_center_frequency'] = obs_info['start_center_frequency']
            info.attrs['channel_bandwidth'] = obs_info['channel_bandwidth']
            info.attrs['nof_beams'] = obs_info['nof_beams']
            info.attrs['nof_channels'] = obs_info['nof_channels']
            info.attrs['pointings'] = obs_info['pointings']
            info.attrs['beam_az_el'] = obs_info['beam_az_el']
            info.attrs['reference_declinations'] = obs_info['declinations']
            info.attrs['nof_subarrays'] = settings.beamformer.nof_subarrays
            info.attrs['calibrated'] = settings.beamformer.calibrate_subarrays
            info.attrs['observation_settings'] = str(obs_info['settings'])

            # Create group that will contain all observation data
            f.create_group('observation_data')

    def _add_beam_data_to_file(self, obs_info, beam_data):
        """ Add beam data to output file"""

        with h5py.File(self._beam_file, mode='a') as f:
            # Create data set name
            dataset_name = f"beam_data"
            timestamp_name = f"beam_timestamp"

            # Load observation data group
            dset = f['observation_data']

            # If data sets do not exist, add them
            if dataset_name not in dset.keys():
                dset.create_dataset(dataset_name,
                                    (0, obs_info['nof_beams'], obs_info['nof_channels']),
                                    maxshape=(None, obs_info['nof_beams'], obs_info['nof_channels']),
                                    chunks=(1, obs_info['nof_beams'], obs_info['nof_channels']),
                                    dtype='f4')

                dset.create_dataset(timestamp_name, 0, maxshape=(None,), dtype='f8')

            # Add data to file
            dset = f[f'observation_data/{dataset_name}']
            dset.resize((dset.shape[0] + obs_info['nof_samples'], dset.shape[1], dset.shape[2]))
            dset[-obs_info['nof_samples']:, :] = beam_data

            # Add timestamp to file
            dset = f[f"observation_data/{timestamp_name}"]
            dset.resize((dset.shape[0] + obs_info['nof_samples'],))
            timestamp = datetime.datetime.timestamp(obs_info['timestamp'])
            dset[-obs_info['nof_samples']:] = timestamp + np.arange(obs_info['nof_samples']) * obs_info['sampling_time']

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._input.shape, datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # If head file not written, write it now
        if self._counter == 0:
            self._create_output_file(obs_info)

        # Save data to output
        if settings.manager.use_gpu:
            output_data[:] = cu.asnumpy(input_data)
        else:
            output_data[:] = input_data[:].copy()

        # Ignore first 2 buffers (because of channeliser)
        self._counter += 1
        if self._counter <= 2:
            return obs_info

        # Perform pre-processing on beam if required
        if 'compute_power' in settings.persister.settings() and settings.persister.compute_power:
            beam_data = np.power(np.abs(output_data), 2)
        else:
            beam_data = output_data

        # Transpose to required shape
        beam_data = np.transpose(beam_data[0, :], (2, 0, 1))

        # Add beam data to output file
        self._add_beam_data_to_file(obs_info, beam_data)

        return obs_info


