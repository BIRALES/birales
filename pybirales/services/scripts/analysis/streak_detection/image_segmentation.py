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
from evaluation import *
from receiver import *

# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed

if __name__ == '__main__':

    SAVE = True
    USE_CACHE = False
    VISUALISE = False
    TRACK_THICKNESS = 1
    N_CHANS = 8192
    N_SAMPLES = 160
    N_TRACKS = 10
    N_TESTS = 10
    ROOT = "/home/denis/.birales/visualisation/fits/bkg_noise_dataset_casa_20190914"
    ROOT = "/home/denis/.birales/visualisation/fits/detection_raw_data"
    EXT = '.pdf'
    # snr = [3, 5, 10]
    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    snr = np.arange(0, 7, 0.5)
    # snr = [2, 3, 5, 10]

    POST_PROCESS = True

    filters = [
        # ('Global Filter', global_thres),
        # ('Local Filter', local_thres),
        # ('Global Filter (R)', global_thres_running),
        # ('Local Filter (R)', local_thres_running),
        ## ('Kittler', kittler),
        # ('Canny', canny_filter),
        # ('sigma_clip_map', sigma_clipping_map),
        ## ('Adaptive Filter', adaptive),
        ## ('CFAR', cfar),

        ('yen', yen),
        ('Otsu Filter', otsu_thres),
        ('iso_data', isodata),
        ('triangle', triangle),
        ('minimum', minimum),
        ('sigma_clip', sigma_clipping)
    ]

    OUTPUT_DF = 'output_' + str(len(filters)) + str(N_CHANS) + str(N_SAMPLES) + str(N_TRACKS) + '_'.join(
        map(str, snr)) + '.pkl'
    metrics_df = pd.DataFrame()
    algorithms = [name for name, _ in filters]
    t1 = time.time()

    if USE_CACHE:
        metrics_df = pd.read_pickle(OUTPUT_DF)
    else:
        np.random.seed(SEED)

        # Create image from real data

        # org_test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=N_CHANS, nsamples=N_SAMPLES)

        # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

        for t in range(0, N_TESTS):
            org_test_img = generate_test_image(ROOT, N_SAMPLES)

            # Remove channels with RFI
            org_test_img = rfi_filter(org_test_img)

            # Estimate the noise
            noise_mean = np.mean(org_test_img)

            # Make a copy of the data such that if it is mutated by the filter, the original is not affected
            test_img = org_test_img.copy()
            # Generate a number of tracks
            tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape, TRACK_THICKNESS)

            # Add tracks to the true data
            true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, 1, .1)

            visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, 1), tracks,
                            visualise=VISUALISE,
                            file_name="true_track.png", bar=False)

            for s in snr:
                print "Evaluating filters with tracks at SNR {:0.2f}W".format(s)

                # Add tracks to the simulated data
                test_img = add_tracks(test_img, tracks, noise_mean, s)

                visualise_image(test_img, 'Test Image: %d tracks at SNR %d dB' % (N_TRACKS, s), tracks,
                                visualise=VISUALISE,  # VISUALISE,
                                file_name="test_image_{}dB.png".format(s))

                # Filter the data in generate metrics
                for f_name, filter_func in filters:
                    data = test_img.copy()
                    # split data in chunks
                    start = time.time()
                    # mask, threshold = chunked_filtering(data, filter_func)
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
                        # mask, timing = morph_opening(data, test_img, mask)
                        mask, _ = hit_and_miss(data, test_img, mask)
                        timing = time.time() - start
                        f_name += '_pp'
                        e2 = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS, f_name)
                        metrics_df = metrics_df.append(e2, ignore_index=True)
                        visualise_filter(test_img, mask, tracks, f_name, s, visualise=VISUALISE,
                                         file_name="filter_{}_{}dB.png".format(f_name, s))

        if SAVE:
            metrics_df.to_pickle(OUTPUT_DF)

    print metrics_df[['name', 'snr', 'dt', 'score', 'recall', 'specificity', 'reduction', 'fpr']].sort_values(
        by=['score', 'recall', 'specificity'], ascending=False)

    print metrics_df[['name', 'snr', 'dt', 'score', 'recall', 'specificity', 'reduction', 'fpr']].sort_values(
        by=['score', 'recall', 'specificity'], ascending=False).groupby(['snr', 'name']).agg(['mean', 'std'])

    print "{} algorithms took {} seconds".format(len(filters), time.time() - t1)

    # metrics_df = metrics_df[metrics_df['snr'] < 11]
    agg_df = metrics_df.groupby(['snr', 'name']).agg(['mean', 'std']).reset_index()

    # visualise the results

    plot_metric_combined(agg_df, ['recall', 'specificity', 'score'],
                         file_name='image_seg_results/combined_v_snr' + EXT)
    plot_metric_combined(agg_df, ['recall', 'specificity', 'score'], pp_results=True,
                         file_name='image_seg_results/combined_pp_v_snr' + EXT)

    plot_timings(agg_df, algorithms=algorithms, include_pp=True, file_name='image_seg_results/timings' + EXT)

    plt.show()
