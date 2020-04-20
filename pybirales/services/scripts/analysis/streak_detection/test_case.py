from detection import *
from filters import *


class ImageSegmentationAlgorithm:
    def __init__(self, name, func):
        self.name = name
        self.func = func


class FeatureExtractionAlgorithm:
    def __init__(self, name, func):
        self.name = name
        self.func = func


class Test:
    def __init__(self, image_seg_algo, detectors):
        self.image_seg_algo = image_seg_algo
        self.detectors = detectors


class TestSuite:
    def __init__(self, name, tests):
        self.name = name
        self.tests = tests


# Not used
gt = ImageSegmentationAlgorithm('Global Filter', global_thres)
lt = ImageSegmentationAlgorithm('Local Filter', local_thres)
gt_r = ImageSegmentationAlgorithm('Global Filter (R)', global_thres_running)
lt_r = ImageSegmentationAlgorithm('Local Filter (R)', local_thres_running)
kittler_filter = ImageSegmentationAlgorithm('Kittler', kittler)
canny = ImageSegmentationAlgorithm('Canny', canny_filter)
sigma_clip_map = ImageSegmentationAlgorithm('sigma_clip_map', sigma_clipping_map)
adaptive_filter = ImageSegmentationAlgorithm('Adaptive Filter', adaptive)

yen_filter = ImageSegmentationAlgorithm('Yen', yen)
otsu_filter = ImageSegmentationAlgorithm('Otsu Filter', otsu_thres)
iso_filter = ImageSegmentationAlgorithm('Iso Data', isodata)
triangle_filter = ImageSegmentationAlgorithm('Triangle', triangle)
sc_3_filter = ImageSegmentationAlgorithm('Sigma Clip', sigma_clipping)
sc_4_filter = ImageSegmentationAlgorithm('sigma_clip4', sigma_clipping4)
no_filter = ImageSegmentationAlgorithm('No filter', dummy_filter)

dbscan_detector = FeatureExtractionAlgorithm('DBSCAN', naive_dbscan)
msds_detector = FeatureExtractionAlgorithm('MSDS', msds_q)
hough_detector = FeatureExtractionAlgorithm('Hough', hough_transform)
astride_detector = FeatureExtractionAlgorithm('Astride', astride)
cfar_detector = FeatureExtractionAlgorithm('CFAR', cfar)

IMAGE_SEG_TESTS = TestSuite('filter_tests', [
    Test(yen_filter, []),
    Test(otsu_filter, []),
    Test(iso_filter, []),
    Test(triangle_filter, []),
    Test(sc_3_filter, []),
    # Test(sc_4_filter, []),
])

DETECTION_TESTS = TestSuite('detection_tests', [
    Test(triangle_filter, [msds_detector, dbscan_detector, hough_detector]),
    Test(sc_3_filter, [msds_detector, dbscan_detector, hough_detector]),
    Test(sc_4_filter, [msds_detector, dbscan_detector, hough_detector]),
    Test(None, [astride_detector, cfar_detector])
])

DETECTION_TESTS_DEBUG = TestSuite('detection_tests', [
    Test(triangle_filter, [msds_detector]),
    # Test(triangle_filter, [msds_detector, dbscan_detector, hough_detector]),
    # Test(no_filter, [astride_detector])
])

DETECTION_ALG = [dbscan_detector, msds_detector, astride_detector, hough_detector]
IS_ALG = [yen_filter, otsu_filter, iso_filter, triangle_filter, sc_3_filter, sc_4_filter]
