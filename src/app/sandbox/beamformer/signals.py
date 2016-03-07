import numpy as np


class Signal():

    @staticmethod
    def step_signal(time):
        out = np.zeros(len(time))
        for count, t in enumerate(time):
            if count > len(time) / 2:
                out[count] = 1
        return out

    @staticmethod
    def get_antenna_input_signal(antennas, time, delays):
        input_signal = []
        for i, antenna in enumerate(antennas):
            f = Signal.step_signal(time)
            f = Signal.buffer_time(int(delays[i]), f, time)
            input_signal.append(f)
        return input_signal

    @staticmethod
    def buffer_time(samples, signal, time):
        buffered = np.lib.pad(signal, (samples, 0), 'constant', constant_values=(0, 0))
        return buffered[:len(time)]
