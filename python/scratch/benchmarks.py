import numpy as np
import timeit

# Define telescope/pipeline parameters
nof_antennas       = 32
nof_beams          = 32
total_nof_channels = 1
channel_bandwidth  = 20e6 / 256.0  # 78.125 kHz coming out of the digital backend
nof_fine_channels  = 16384   # To get 4.7 Hz resolution
bits_per_sample    = 32

# Test parameters
nof_iterations = 100

def benchmark_fft():
    """ Benchmark FFT performance """
    global data, output
    for i in xrange(nof_antennas):
        output[i, :] = np.fft.fft(data[i, :])

def benchmark_beamformer():
    """ Benchmark beamformer performance """
    global data, weights, output
    output = np.sum(data * weights, axis=1)


if __name__ == "__main__":
    # Global data handle to prepare data for methods
    global data, weights, output

    # Benchmark FFT
    data = np.ones((nof_antennas, nof_fine_channels), dtype=complex)
    output = np.zeros((nof_antennas, nof_fine_channels), dtype=complex)
    exec_time = timeit.timeit("benchmark_fft()", 'from __main__ import benchmark_fft', number=nof_iterations)
    print exec_time
    seconds_of_data = (nof_iterations * nof_fine_channels / channel_bandwidth)
    print "FFT executes at %.4f%% real time (equivalent to ~%.2f MB/s)" % \
          (exec_time / seconds_of_data, seconds_of_data * nof_antennas * channel_bandwidth * bits_per_sample / (1024 * 8192.0))

    # Benchmark beamformer
    weights = np.ones(nof_antennas, dtype=complex)
    data = np.ones((nof_fine_channels, nof_antennas), dtype=complex)
    output = np.zeros(nof_fine_channels, dtype=complex)
    exec_time = timeit.timeit("benchmark_beamformer()", 'from __main__ import benchmark_beamformer', number=nof_iterations)
    seconds_of_data = (nof_iterations * nof_fine_channels / channel_bandwidth)
    print "Beamformer executes at %.4f%% real time (equivalent to ~%.2f MB/s)" % \
          (exec_time / seconds_of_data, seconds_of_data * nof_antennas * channel_bandwidth * bits_per_sample / (1024 * 8192.0))
