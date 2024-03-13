import os

import h5py


def print_beam_hdf5_information(h5_filename, show_settings=False):
    """ Print metadata from the beam data hdf5 file """
    with h5py.File(h5_filename, 'r') as f:
        info = f['observation_info']
        print(f"Observation name: {info.attrs['observation_name']}")
        print(f"Transmitter frequency: {info.attrs['transmitter_frequency']}")
        print(f"Sampling time (s): {info.attrs['sampling_time']}")
        print(f"Start center frequency (Hz): {info.attrs['start_center_frequency']}")
        print(f"Channel bandwidth (Hz): {info.attrs['channel_bandwidth']}")
        print(f"Number of beams: {info.attrs['nof_beams']}")
        print(f"Number of channels: {info.attrs['nof_channels']}")

        dset = f['observation_data/beam_data']
        nof_samples = dset.shape[0]

        print(f"Number of samples: {nof_samples}")
        print(f"Number of subarrays: {info.attrs['nof_subarrays']}")
        print(f"Calibrated observation: {info.attrs['calibrated']}")
        print(f"Reference declinations: {info.attrs['reference_declinations']}")
        print(f"Pointings: {info.attrs['pointings']}")

        if show_settings:
            print(info.attrs['observation_settings'])


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="Usage: %prog [options] arguments")
    parser.add_option("-f", "--input-file", dest="input_file", default=None,
                      help="Input file to plot (Required)")
    parser.add_option("-s", "--show-settings", dest="settings", default=False, action="store_true",
                      help="Display observation settings (default: False)")
    (options, args) = parser.parse_args()

    # Sanity checks
    if options.input_file is None:
        print("Please specify an input file")
        exit()
    elif not os.path.exists(options.input_file):
        print("Specified input file does not exist")
        exit()

    print_beam_hdf5_information(options.input_file, show_settings=options.settings)
