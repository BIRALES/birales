from datetime import datetime

from helpers import DateTimeHelper


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
        self.id = self.beam.observation_name + '.' + str(self.beam.id) + '.' + str(name)
        self.detection_data = detection_data

        self.data = {
            'time': [],
            'mdj2000': [],
            'time_elapsed': [],
            'frequency': [],
            'doppler_shift': [],
            'snr': [],
        }

        self.detections = []

        # todo - This function can be called lazily, since it is only for visualisation
        self.set_data(detection_data)

    def set_data(self, detection_data):
        for frequency, time, snr in sorted(detection_data, key=lambda row: row[1]):
            self.data['time'].append(time)
            self.data['mdj2000'].append(self._get_mdj2000(time))
            self.data['time_elapsed'].append(self._time_elapsed(time))
            self.data['frequency'].append(frequency)
            self.data['doppler_shift'].append(self._get_doppler_shift(self.beam.tx, frequency))
            self.data['snr'].append(snr)

            self.detections.append({
                'time': time,
                'mdj2000': self._get_mdj2000(time),
                'time_elapsed': self._time_elapsed(time),
                'frequency': frequency,
                'doppler_shift': self._get_doppler_shift(self.beam.tx, frequency),
                'snr': snr,
            })

    @staticmethod
    def _get_doppler_shift(transmission_frequency, reflected_frequency):
        return (reflected_frequency - transmission_frequency) * 1e6

    @staticmethod
    def _time_elapsed(time):
        return 'n/a'

    @staticmethod
    def _get_mdj2000(dates):
        return DateTimeHelper.ephem2juldate(dates)

    def __iter__(self):
        yield '_id', self.id
        yield 'name', self.name
        yield 'data', self.data
        yield 'detections', self.detections
        yield 'beam_id', self.beam.id
        yield 'data_set_id', self.beam.data_set.id
        yield 'created_at', datetime.now()
