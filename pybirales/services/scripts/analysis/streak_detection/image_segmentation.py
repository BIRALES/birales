"""
This script evaluates the performance of different filters for
image segmentation of the spectrogram produced by the PyBirales
Channeliser.

Notebook reproduces the figures used in the
streak detection paper.

"""

import pandas as pd

pd.options.display.max_columns = 20
pd.set_option('display.width', 4000)
pd.set_option('precision', 4)

from detection import *
import time
from evaluation import evaluate_detector, evaluate_filter
from filters import hit_and_miss
from receiver import get_test_tracks, generate_test_image, add_tracks
from test_case import IMAGE_SEG_TESTS, DETECTION_ALG
from configuration import GRADIENT_RANGE, TRACK_LENGTH_RANGE
from pybirales.pipeline.modules.detection.msds.util import snr_calc, is_valid
from pybirales.pipeline.modules.detection.msds.visualisation import *


# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed

def test_detector(detector, data, true_tracks, noise_estimate, debug=True, visualise=False):
    print "Running {} algorithm".format(detector.name)
    start = time.time()
    candidates = detector.func(data, true_tracks, noise_estimate, debug)

    candidates = [snr_calc(candidate, noise_estimate=noise_mean) for candidate in candidates if is_valid(candidate)]
    timing = time.time() - start
    print "The {} detection algorithm, found {} candidates in {:2.3f}s".format(detector.name, len(candidates), timing)

    visualise_post_processed_tracks(candidates, true_tracks, '6_post-processed-tracks.png', limits=None,
                                    groups=None,
                                    debug=False)

    # visualise candidates
    # visualise_detector(data, candidates, tracks, detector.name, s, visualise=visualise)

    return evaluate_detector(true_image, data, candidates, timing, s, thickness=TRACK_THICKNESS, name=detector.name)


def show_results_filter(metrics_df, im_algorithms, directory):
    # print metrics_df[['name', 'snr', 'dt', 'score', 'recall', 'specificity', 'reduction', 'fpr']].sort_values(
    #     by=['score', 'recall', 'specificity'], ascending=False)
    #
    # print metrics_df[['name', 'snr', 'dt', 'score', 'recall', 'specificity', 'reduction', 'fpr']].sort_values(
    #     by=['score', 'recall', 'specificity'], ascending=False).groupby(['snr', 'name']).agg(['mean', 'std'])

    print "{} tests took {} seconds".format(len(TEST_SUITE.tests), time.time() - t1)

    agg_df = metrics_df.groupby(['snr', 'name']).agg(['mean', 'std']).reset_index()

    n_algs = len(im_algorithms)

    file_name = 'SNR={}-{}dB__ALG={}'.format(agg_df['snr'].min(), agg_df['snr'].max(), n_algs)

    # visualise the results
    plot_metric_combined(agg_df, ['recall', 'specificity', 'score'],
                         file_name=directory + '/' + file_name + '__combined_v_snr' + EXT)
    plot_metric_combined(agg_df, ['recall', 'specificity', 'score'], pp_results=True,
                         file_name=directory + '/' + file_name + '__combined_pp_v_snr' + EXT)

    plot_timings(agg_df, algorithms=im_algorithms, include_pp=True,
                 file_name=directory + '/' + file_name + '__timings' + EXT)

    print agg_df[['name', 'snr', 'dt', 'score', 'recall', 'specificity', 'reduction', 'fpr']]

    agg_df.to_csv(directory + '/' + file_name + '__export.csv')

    return agg_df


def str_join(params):
    params = [str(p) for p in params]
    return '_'.join(params)


def show_results_detector(metrics_df, algorithms, directory):
    # print metrics_df[['name', 'snr', 'dt', 'recall', 'precision', 'f1']]
    agg_df = metrics_df.groupby(['snr', 'name']).agg(['mean', 'std']).reset_index()

    # snrs = agg_df['snr'].unique()
    n_algs = len(algorithms)

    file_name = 'SNR={}-{}dB__ALG={}'.format(agg_df['snr'].min(), agg_df['snr'].max(), n_algs)

    # visualise the results
    plot_metric_combined(agg_df, ['recall', 'precision', 'f1'],
                         file_name=directory + '/' + file_name + '__combined_v_snr' + EXT)

    plot_timings_detector(agg_df, algorithms=algorithms, file_name=directory + '/' + file_name + '__timings' + EXT)

    print agg_df[['name', 'snr', 'dt', 'recall', 'precision', 'f1']].sort_values(by=['snr'])

    agg_df.to_csv(directory + '/' + file_name + '__export.csv')

    return agg_df


if __name__ == '__main__':
    SEED = 5681002
    np.random.seed(SEED)
    random.seed(SEED)

    SAVE = True
    USE_CACHE = False
    VISUALISE = False

    POST_PROCESS = True
    TRACK_THICKNESS = 2
    N_CHANS = 4110
    N_SAMPLES = 160
    N_TRACKS = 5

    ROOT = "/home/denis/.birales/visualisation/fits/bkg_noise_dataset_casa_20190914"
    ROOT = "/home/denis/.birales/visualisation/fits/detection_raw_data"
    EXT = '.pdf'
    TEST_SUITE = IMAGE_SEG_TESTS
    # TEST_SUITE = DETECTION_TESTS
    # TEST_SUITE = DETECTION_TESTS_DEBUG

    im_algorithms = [t.image_seg_algo.name for t in TEST_SUITE.tests]
    detection_algorithms = [d.name for d in DETECTION_ALG]
    metrics_df = pd.DataFrame()
    detection_metrics_df = pd.DataFrame()

    snr = np.arange(0, 6, 0.5)
    snr = np.arange(0, 7.25, 0.25)

    # snr = np.arange(0, 7.5, 1)
    # snr = np.arange(0, 10, 2)
    snr = [1]
    # snr = [5, 10, 15]
    # snr = [2, 3, 5, 8, 10, 15]
    # snr = [0.5, 1, 2]
    # snr = [1, 5, 10]
    # snr = [5]
    snr = np.arange(0, 6.5, 0.5)
    DEBUG_DETECTOR = False
    N_TESTS = 10
    VISUALISE = False

    OUTPUT_DF = 'output_' + str(len(TEST_SUITE.name)) + str(N_CHANS) + str(N_SAMPLES) + str(N_TRACKS) + '_'.join(
        map(str, snr)) + '.pkl'

    # Create image from real data
    # org_test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=N_CHANS, nsamples=N_SAMPLES)

    # visualise_image(test_img, 'Test Image: no tracks', tracks=None)
    count = 0
    t1 = time.time()
    for t in range(0, N_TESTS):
        # if t < 4:
        #     continue
        org_test_img = generate_test_image(ROOT, N_SAMPLES)

        # Estimate the noise
        noise_mean = np.mean(org_test_img)

        # Generate a number of tracks
        tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, org_test_img.shape, TRACK_THICKNESS)

        # tracks = get_test_tracks_crossing(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, org_test_img.shape, TRACK_THICKNESS)

        # Add tracks to the true data

        true_image = add_tracks(np.zeros(shape=org_test_img.shape), tracks, 1, .1)

        visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, 1), tracks,
                        visualise=VISUALISE,
                        file_name="true_track.png", bar=False)
        #
        # if t < 3:
        #     continue
        #
        # if t == 4:
        #     DEBUG_DETECTOR = True
        # else:
        #     DEBUG_DETECTOR = False

        for s in snr:
            print "Evaluating filters with tracks at SNR {:0.2f} dB and noise {:0.3f}".format(s, noise_mean)

            # Add tracks to the simulated data
            test_img = add_tracks(org_test_img.copy(), tracks, noise_mean, s)

            # preprocessing - remove noise estimate from image
            # test_img -= noise_mean
            test_img -= np.mean(test_img, axis=1)[..., np.newaxis]

            visualise_image(test_img, 'Test Image: %d tracks at SNR %d dB' % (N_TRACKS, s), tracks,
                            visualise=VISUALISE,  # VISUALISE,
                            file_name="test_image_{}dB.png".format(s))
            # continue
            # Filter the data in generate metrics
            for test in TEST_SUITE.tests:
                data = test_img.copy()
                # split data in chunks
                start = time.time()

                test_filter = test.image_seg_algo
                f_name = test_filter.name
                filter_func = test_filter.func

                try:
                    mask, threshold = filter_func(data)
                except Exception:
                    print "warning. The following filter failed: ", f_name, "at an snr of ", s
                    continue
                timing = time.time() - start

                # Visualise filter
                e = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS, f_name)
                metrics_df = metrics_df.append(e, ignore_index=True)
                visualise_filter(test_img, mask, tracks, f_name, s, threshold, visualise=VISUALISE,
                                 file_name="filter_{}_{}dB.png".format(f_name, s))

                # Filter post-processor
                if POST_PROCESS:
                    start = time.time()
                    mask, _ = hit_and_miss(data, test_img, mask)
                    timing = time.time() - start
                    f_name += '_pp'
                    e2 = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS, f_name)
                    metrics_df = metrics_df.append(e2, ignore_index=True)
                    visualise_filter(test_img, mask, tracks, f_name, s, visualise=VISUALISE,
                                     file_name="filter_{}_{}dB.png".format(f_name, s))

                data[mask] = 0

                # Detection stuff here
                for detector in test.detectors:
                    data2 = data.copy()
                    # print 'dd',  detector.name, np.mean(data2), np.mean(test_img), snr, s
                    results = test_detector(detector, data2, tracks, noise_mean, debug=DEBUG_DETECTOR)
                    detection_metrics_df = detection_metrics_df.append(results, ignore_index=True)

    agg_df = show_results_filter(metrics_df, im_algorithms, 'image_seg_results')

    # agg_df = show_results_detector(detection_metrics_df, detection_algorithms, 'detection_results')

    plt.show()
