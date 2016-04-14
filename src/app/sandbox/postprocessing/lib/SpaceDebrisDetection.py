from abc import abstractmethod
from app.sandbox.postprocessing.lib.SpaceDebrisCandidateCollection import SpaceDebrisCandidateCollection


class SpaceDebrisDetection(object):
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def detect(self, beam):
        detections = self.detection_strategy.detect(beam)
        candidates = SpaceDebrisCandidateCollection(candidates = detections)
        return candidates


class SpaceDebrisDetectionStrategy(object):
    def __init__(self, max_detections = 10):
        self.max_detections = max_detections  # maximum detections per beam

    @abstractmethod
    def detect(self, beam):
        pass


