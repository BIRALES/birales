import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as CONSTANT
from scipy import signal


def antenna_positions(n):
    antenna_pos = {
        1: (0, 0),
        2: (10, 0),
        3: (50, 0),
    }
    antenna_pos = np.matrix([0, 10, 50])
    antenna_pos = np.linspace(0, 10, n)
    return antenna_pos


def plot_signals(data_x, data_y, antennas):
    n = len(antennas)
    a = np.floor(n ** 0.5).astype(int)
    b = np.ceil(1. * n / a).astype(int)

    fig = plt.figure(figsize=(3. * b, 3. * n))
    for i in range(0, n):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.set_xlim([0, 0.01])
        ax.set_ylim([-1.5, 1.5])
        ax.set_ylabel('S' + str(i + 1) + ' Amplitude')
        ax.grid(True)
        ax.plot(data_x, data_y[:, i])
    plt.xlabel('Time (s)')
    plt.show()


def plot_signal(data_x, data_y, axis_label_x, axis_label_y, title, show=True):
    plt.xlabel(axis_label_x)
    plt.ylabel(axis_label_y)
    plt.title(title)
    plt.xlim([0, 0.01])
    plt.ylim([-1.5, 1.5])
    plt.plot(data_x, data_y)
    if show:
        plt.show()


def sub_plot(data_x, data_y, axis_label_x, axis_label_y, title, n, position, show=True):
    plt.subplot(n, 1, position)
    plot_signal(data_x, data_y, axis_label_x, axis_label_y, title, show)


def calculate_delays(distances, speed, signal_angle):
    t = np.matrix(distances / speed)
    td = np.matrix(np.cos(signal_angle) * t)
    return td


def mock_signal_data(samples, sampling_rate, signal_angle, speed, frequency, antennas):
    time = np.linspace(0, samples / sampling_rate, samples, endpoint=False)
    i_signal = input_signal(time, frequency)
    time_delays = calculate_delays(antennas, speed, signal_angle)

    """ wtf """
    fft_freqs = np.matrix(np.linspace(0, sampling_rate, samples, endpoint=False))
    spacial_filt = spacial_filter(fft_freqs, time_delays)
    replicas = np.fft.irfft(np.array(np.fft.fft(i_signal)) * spacial_filt, samples, 1)
    # add to data and then return it
    data = replicas.transpose()
    """ wtf """

    return data, time


def spacial_filter(fft_frequncies, time_delays):
    filter = np.exp(2j * np.pi * fft_frequncies.transpose() * time_delays)
    filter = np.array(filter).transpose()
    return filter


def input_signal(time, frequency):
    f = signal.square(2 * np.pi * frequency * time)
    # i, q, e = signal.gausspulse(time, fc=5, retquad=True, retenv=True)
    f = np.sqrt(1) * np.random.randn(10000)
    f = np.sqrt(1) * np.random.randn(SAMPLES)
    return f


def cbf(data, samples, sampling_rate, speed, antenna_positions, angle):
    look_dirs = np.arccos(np.linspace(-1, 1, 180))
    n = len(antenna_positions)
    bf_data = np.zeros((samples, len(look_dirs)))
    # find time lags between phones and the bf matrix
    time_delays = np.matrix((antenna_positions / speed))
    fft_freqs = np.matrix(np.linspace(0, sampling_rate, samples, endpoint=False)).transpose()

    for ind, direction in enumerate(look_dirs):
        spacial_filt = 1.0 / n * np.exp(-2j * np.pi * fft_freqs * time_delays * np.cos(direction))
        # fft the data, and let's beamform.
        bf_data[:, ind] = np.sum(np.fft.irfft(np.fft.fft(data, samples, 0) * np.array(spacial_filt), samples, 0), 1)

    return bf_data


def calculate_signal_power(data, samples):
    power = sum(abs(np.fft.fft(bf_data, samples, 0)) ** 2 / samples ** 2, 0)
    return power


def directions():
    return np.arccos(np.linspace(-1, 1, 180))


def plot_antenna_directionality(data, samples):
    look_dirs = directions()
    power = calculate_signal_power(data, samples)

    ax = plt.subplot(111, projection='polar')
    ax.plot(look_dirs, power, '-')
    ax.set_rmax(2.0)
    ax.grid(True)
    plt.show()


def plot_antenna_power(data, time, samples, axis_label_x, axis_label_y, title):
    power = calculate_signal_power(data, samples)
    plt.xlabel(axis_label_x)
    plt.ylabel(axis_label_y)
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot(power, '*-')
    plt.show()


def plot_beam_power(data, directions, samples, axis_label_x, axis_label_y, title):
    power = calculate_signal_power(data, samples)
    plt.xlabel(axis_label_x)
    plt.ylabel(axis_label_y)
    plt.title(title)
    plt.plot(directions, power, '*-')
    plt.show()


SAMPLES = 10000
SAMPLING_RATE = 10000
SIGNAL_ANGLE = np.pi / 4
SPEED = 200
FREQUENCY = 500
N_ANTENNAS = 32
ANTENNAS = antenna_positions(N_ANTENNAS)
signal_data, time = mock_signal_data(SAMPLES, SAMPLING_RATE, SIGNAL_ANGLE, SPEED, FREQUENCY, ANTENNAS)
bf_data = cbf(signal_data, SAMPLES, SAMPLING_RATE, SPEED, ANTENNAS, SIGNAL_ANGLE)

# Original Signal Received by N Antennas


# Original Power in Antennas
plot_antenna_power(signal_data, time, SAMPLES, title='Power received at Antennas', axis_label_x='Time',
                   axis_label_y='Power')

# Plot Power in each beam
plot_beam_power(bf_data, directions(), SAMPLES, title='Power in each Beam', axis_label_x='Beam',
                axis_label_y='Power')

# Beam direction
plot_antenna_directionality(bf_data, SAMPLES)
