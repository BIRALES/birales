import os.path

from matplotlib import pyplot as plt
import numpy as np
import h5py


# Set matplotlib settings
import matplotlib.style as style
style.use('tableau-colorblind10')   # Color-blind friendly
plt.set_cmap('viridis')             # Color-blind friendly

plt.rcParams['legend.handlelength'] = 5.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14


class BeamPlotter:
    """ Class that read and plots data from beam hdf5 files """

    def __init__(self, filepath, beam=0, start_channel=0,
                 stop_channel=-1, integrate=1, log=False, start_sample=0,
                 nof_samples=-1):
        """ Class initializer """

        # Define and initialise
        self._filepath = filepath
        self._beam = beam
        self._start_channel = start_channel
        self._stop_channel = stop_channel
        self._integrate = integrate
        self._log = log
        self._start_sample = start_sample
        self._nof_samples = nof_samples

        # Load file metadata
        self._metadata = self.load_metadata()

        # Get number of spectra in file
        self._nof_spectra_in_file = self.get_nof_spectra_in_file()

        # Fix number of samples and stop channel if -1 was specified
        if self._stop_channel == -1:
            self._stop_channel = self._metadata['nof_channels']
        if self._nof_samples == -1:
            self._nof_samples = (self._nof_spectra_in_file - self._start_sample) // self._integrate
        self._nof_channels = self._stop_channel - self._start_channel

        # Sanity check on parameters
        if beam < 0 or beam > self._metadata['nof_beams']:
            raise Exception(f"Invalid beam number specified. Must be between 0 and {self._metadata['nof_beams']}")
        if self._start_channel > self._metadata['nof_channels']:
            raise Exception(f"Invalid start channel specified. Must be between 0 and {self._metadata['nof_channels']}")
        if self._stop_channel > self._metadata['nof_channels']:
            raise Exception(f"Invalid stop channel specified. Must be between 0 and {self._metadata['nof_channels']}")
        if self._start_sample > self._nof_spectra_in_file:
            raise Exception(f"Invalid start sample specified. Must be between 0 and {self._nof_spectra_in_file}")
        if self._start_sample + self._integrate * self._nof_samples > self._nof_spectra_in_file:
            raise Exception(f"Number of samples exceeds end of file (file contains {self._nof_spectra_in_file} spectra")

        # Generate frequencies for specified parameters
        self._frequencies = (self._metadata['start_center_frequency'] +
                             start_channel * self._metadata['channel_bandwidth'] +
                             np.arange(self._nof_channels) * self._metadata['channel_bandwidth'])

        # Load timestamps
        self._timestamps = self.load_timestamps()

        # Beam pointing
        self._pointing = self._metadata['pointings'][self._beam]
        self._pointing[1] += self._metadata['reference_declinations'][0]

    def load_metadata(self):
        """ Load file metadata """
        with h5py.File(self._filepath, 'r') as f:
            return {k: v for k, v in f['observation_info'].attrs.items()}

    def load_timestamps(self):
        """ Load timestamps for required range """
        with h5py.File(self._filepath, 'r') as f:
            dset = f['observation_data']['beam_timestamp']
            timestamps = dset[self._start_sample: self._start_sample + self._integrate * self._nof_samples]
            return timestamps[::self._integrate] - timestamps[0]

    def get_nof_spectra_in_file(self):
        """ Get the number of spectra in the file """
        with h5py.File(self._filepath, 'r') as f:
            dset = f['observation_data']['beam_data']
            return dset.shape[0]

    def plot_transit(self):
        """ Generate a transit plot """
        with h5py.File(self._filepath, 'r') as f:
            # Read data from file
            data = f['observation_data/beam_data']

            # Get required range
            data = data[self._start_sample:self._start_sample + self._integrate * self._nof_samples,
                        self._beam, self._start_channel:self._stop_channel]

            # Sum across frequencies. Data is now summed transit profile
            data = np.sum(data.squeeze(), 1)

            # If required, integrate in time
            if self._integrate:
                data = np.mean(data.reshape((self._nof_samples, self._integrate)), axis=1)

            # If required, compute logarithm
            if self._log:
                data = 10 * np.log10(data)

            # Plot data
            plt.plot(self._timestamps, data)
            plt.title(f"Beam {self._beam} (HA: {self._pointing[0]}, DEC: {self._pointing[1]})")
            plt.xlabel("Time since start (s)")
            plt.ylabel("Arbitrary power" + " (log)" if self._log else "")
            plt.show()

    def plot_waterfall(self):
        """ Generate a waterfall plot """
        with h5py.File(self._filepath, 'r') as f:
            # Read data from file
            data = f['observation_data/beam_data']

            # Get required range
            data = data[self._start_sample:self._start_sample + self._integrate * self._nof_samples,
                        self._beam, self._start_channel:self._stop_channel]

            # If required, integrate in time
            if self._integrate:
                data = np.mean(data.reshape((self._nof_samples, self._integrate, self._nof_channels)), axis=1)

            # If required, compute logarithm
            if self._log:
                data = 10 * np.log10(data)

            # Plot data
            plt.imshow(data, aspect='auto', extent=[self._frequencies[0], self._frequencies[-1],
                                                    self._timestamps[0], self._timestamps[-1]])
            plt.title(f"Beam {self._beam} (HA: {self._pointing[0]}, DEC: {self._pointing[1]})")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Time since start (s)")
            cbar = plt.colorbar()
            cbar.set_label("Arbitrary power" + " (log)" if self._log else "", rotation=270, labelpad=20)
            plt.show()


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser(usage="Usage: %prog [options] arguments")
    parser.add_option("-f", "--input-file", dest="input_file", default=None,
                      help="Input file to plot (Required)")
    parser.add_option("-b", "--beam", dest="beam", default=0, type=int,
                      help="Beam to plot (default: 0")
    parser.add_option("-c", "--start-channel", dest="start_channel", default=0, type=int,
                      help="Start channel to plot (default: 0")
    parser.add_option("-e", "--stop-channel", dest="stop_channel", default=-1, type=int,
                      help="Stop channel to plot (default: -1, last channel")
    parser.add_option("-i", "--integrate", dest="integrate", default=1, type=int,
                      help="Samples to integrate (default: 1")
    parser.add_option("--log", dest="log", default=False, action="store_true",
                      help="Plot logarithmic (default: False)")
    parser.add_option("-s", "--start-sample", dest="start_sample", default=0, type=int,
                      help="Start sample to plot (default: 0)")
    parser.add_option("-n", "--nof-samples", dest="nof_samples", default=-1, type=int,
                      help="Number of samples to plot (default: -1, till the end of the file). "
                           "When integrating, this is the number of integrated samples")
    parser.add_option("--transit", dest="transit", default=False, action="store_true",
                      help="Show transit plots, summing across frequencies (default: False)")
    parser.add_option("--waterfall", dest="waterfall", default=False, action="store_true",
                      help="Waterfall plot. Only one beam can be selected (default: False)")
    (options, args) = parser.parse_args()

    # Sanity checks
    if options.input_file is None:
        print("Please specify an input file")
        exit()
    elif not os.path.exists(options.input_file):
        print("Specified input file does not exist")
        exit()

    if not (options.transit or options.waterfall):
        print("Plotting mode (transit or waterfall) must be specified")
        exit()

    # Initialise plotter
    plotter = BeamPlotter(options.input_file, options.beam, options.start_channel,
                          options.stop_channel, options.integrate, options.log,
                          options.start_sample, options.nof_samples)

    if options.transit:
        plotter.plot_transit()
    elif options.waterfall:
        plotter.plot_waterfall()
