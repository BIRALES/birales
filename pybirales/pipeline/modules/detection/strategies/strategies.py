import logging as log
import sys
from abc import abstractmethod
from pybirales.base import settings


class SpaceDebrisDetection(object):
    name = None

    def __init__(self, detection_strategy):
        try:
            self.detection_strategy = detection_strategy
            self.name = self.detection_strategy.name
        except KeyError:
            log.error('%s is not a valid detection strategy. Exiting.', self.detection_strategy)
            sys.exit()

    def detect(self, obs_info, input_data, pool, queue):
        return self.detection_strategy.detect(obs_info, input_data, pool, queue)


class SpaceDebrisDetectionStrategy(object):
    def __init__(self):
        self.max_detections = settings.detection.max_detections

    @abstractmethod
    def detect(self, obs_info, input_data, pool, queue):
        pass

    # def pre_process(self):
    #     pass
    #
    # def post_process(self):
    #     pass
    #
    # def process(self):
    #     pass
    #
    # def multi_threaded(self):
    #     pass
    #
    # def multi_processing(self):
    #     pass
    #
    # def single_threaded(self):
    #     pass
