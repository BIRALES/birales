import numpy as np
import scipy.signal as sp
from signals import Signal
from post_processing import PostProcessing as pp


class TimeDelayAndSumBeamformer:
    N_TAPS = 11
    N = 7 # Number of Antennas
    Hz = 4000.0  # Signal frequency in Hz
    c = 343.3

    ANGLE = np.deg2rad(45)
    SAMPLING_RATE = 8000.0
    SAMPLES = 400
    ANTENNAS = np.linspace(0, 1, N, endpoint=False)
    TIME = np.linspace(0, 100 * Hz, SAMPLES)

    def __init__(self):
        delays = self.get_delays()
        self.INPUT_SIGNALS = Signal.get_antenna_input_signal(self.ANTENNAS, self.TIME, delays)

    @staticmethod
    def hamming(x, n_taps):
        window = 0.54 - 0.46 * np.cos(2.0 * np.pi * (x + 0.5) / n_taps)
        return window

    @staticmethod
    def calculate_weights(delay, n_taps):
        centre_tap = 0.5 * n_taps
        tap_weights = []
        for tap in range(0, n_taps):
            x_shift = tap - delay
            x = np.pi * (x_shift - centre_tap)
            sinc = np.sin(x) / x

            # Hamming windowing function
            window = TimeDelayAndSumBeamformer.hamming(x_shift, n_taps)

            # Calculate the tap weight
            tap_weights.append(window * sinc)
        return tap_weights

    @staticmethod
    def calculate_delay(distance, sampling_rate, velocity, angle):
        """
        :param distance: Distance from the reference antenna
        :param sampling_rate: The sampling rate
        :param velocity: The velocity of the signal
        :param angle: The angle of incidence

        :return delay: Number of samples to delay by
        """
        delay = (sampling_rate * distance * np.cos(angle)) / velocity

        return delay  # number of samples to delay

    def delay_and_sum(self):
        bf_data = []
        for count, da in enumerate(self.ANTENNAS):
            input_signal = self.INPUT_SIGNALS[count]
            sd = self.get_antenna_delay(count)
            # print 'Signal from Antenna', count, 'will be delayed by', round(sd, 3), 'samples'
            fractional = sd - np.fix(sd)
            ws = TimeDelayAndSumBeamformer.calculate_weights(fractional, self.N_TAPS)
            bf = sp.lfilter(ws, 1.0, input_signal)  # ??

            bf_data.append(TimeDelayAndSumBeamformer.buffer_time(int(sd), bf, self.TIME))

        return bf_data

    def generate_beam_pattern(self):
        out = []
        out_log = []
        angles = np.linspace(np.pi / -2., np.pi / 2., 180)

        corrections = []
        for antenna_id, antenna in enumerate(self.ANTENNAS):
            corrections.append(
                self.get_antenna_delay(antenna_id) / gejself.SAMPLING_RATE)

        for angle in angles:
            distances = self.ANTENNAS * np.sin(angle)
            time_delays = np.matrix((distances / self.c))

            time_delays = time_delays + np.matrix(corrections)

            r_sum = np.sum(np.cos(2.0 * np.pi * self.Hz * time_delays))
            i_sum = np.sum(np.sin(2.0 * np.pi * self.Hz * time_delays))

            o = np.sqrt(r_sum ** 2 + i_sum ** 2) / self.N
            o_log = 20.0 * np.log10(o)

            if o_log < -50:
                o_log = -50

            out.append(o)
            out_log.append(o_log)
        return angles, out, out_log

    def get_delays(self):
        delays = []
        for i, antenna_distance in enumerate(self.ANTENNAS):
            delay = TimeDelayAndSumBeamformer.calculate_delay(antenna_distance, self.SAMPLING_RATE, self.c, self.ANGLE)
            delays.append(delay)
        return delays

    def get_antenna_delay(self, antenna_id):
        delays = []
        antenna_delay = 0.
        for i, antenna_distance in enumerate(self.ANTENNAS):
            delay = TimeDelayAndSumBeamformer.calculate_delay(antenna_distance, self.SAMPLING_RATE, self.c, self.ANGLE)
            delays.append(delay)

            if i is antenna_id:
                antenna_delay = delay
        max_delay = max(delays)
        return max_delay - antenna_delay

    @staticmethod
    def beamform(bf_data):
        summation = np.zeros(len(bf_data[0]))
        for signal in bf_data:
            summation = summation + signal

        return summation

    @staticmethod
    def buffer_time(samples, signal, time):
        buffered = np.lib.pad(signal, (samples, 0), 'constant', constant_values=(0, 0))
        return buffered[:len(time)]

    @staticmethod
    def calculate_side_lobes_position(c, f, l, n):
        lobes = range(0, n)
        for n in lobes:
            arg = c / (f * l)
            theta = np.arcsin(n * arg)
            print np.rad2deg(theta)


bf = TimeDelayAndSumBeamformer()
bf_data = bf.delay_and_sum()

beam_pattern = bf.generate_beam_pattern()
pp.plot_beam_pattern(beam_pattern, 'TDAS')
pp.plot_results(bf, bf_data, 'TDAS')
