import numpy as np
from datetime import datetime
from astropy.time import Time


class SpaceDebrisCandidate:
    """
    This class encapsulates the space debris detection as detected across multiple beams

    """

    # todo - merge beam space debris candidates into one
    def __init__(self):
        self.beam_space_debris = []

    def add(self, beam_space_debris):
        self.beam_space_debris.append(beam_space_debris)


class BeamSpaceDebrisCandidate:
    """
    This class encapsulates a space debris candidate detected in a beam
    """

    def __init__(self, name, beam, detection_data):
        self.beam = beam
        self.name = name
        self.id = self.beam.observation_name + '.' + self.beam.data_set.name + '.' + str(self.beam.id) + '.' + str(name)
        self.detections = []
        self.illumination_time = np.min(detection_data[:, 1])  # get minimum time

        # todo - This function can be called lazily, since it is only for visualisation
        self.set_data(detection_data)

    def set_data(self, detection_data):
        for frequency, elapsed_time, snr in detection_data:
            self.detections.append({
                'time': self._timestamp(elapsed_time),
                'mdj2000': self._get_mjd2000(elapsed_time),
                'time_elapsed': self._time_elapsed(elapsed_time),
                'frequency': frequency,
                'doppler_shift': self._get_doppler_shift(self.beam.tx, frequency),
                'snr': snr,
            })

    @staticmethod
    def _get_doppler_shift(transmission_frequency, reflected_frequency):
        return (reflected_frequency - transmission_frequency) * 1e6

    @staticmethod
    def _time_elapsed(elapsed_time):
        return elapsed_time

    def _time(self, time):
        ref_time = self.beam.data_set.config['timestamp'] / 1000.

        return Time(time + ref_time, format='unix')

    def _timestamp(self, elapsed_time):
        time = self._time(elapsed_time)
        return time.iso

    def _get_mjd2000(self, elapsed_time):
        time = self._time(elapsed_time)
        return time.mjd

    def __iter__(self):
        yield '_id', self.id
        yield 'name', self.name
        yield 'detections', self.detections
        yield 'beam_id', self.beam.id
        yield 'data_set_id', self.beam.data_set.id
        yield 'illumination_time', self.illumination_time
        yield 'created_at', datetime.now()
