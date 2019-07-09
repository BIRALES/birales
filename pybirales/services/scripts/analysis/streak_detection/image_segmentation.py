"""
This script evaluates the performance of different filters for
image segmentation of the spectrogram produced by the PyBirales
Channeliser.

Notebook reproduces the figures used in the
streak detection paper.

"""

import time

import configuration
from detection import *
from evaluation import *
from receiver import *
from visualisation import *

# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed

if __name__ == '__main__':

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [10]
    metrics = {}
    metrics_detector = {}
    metrics_df = pd.DataFrame()
    metrics_detector_df = pd.DataFrame()
    filters = [
        # ('Global Filter', global_thres),
        # ('Local Filter', local_thres),
        # ('Global Filter (R)', global_thres_running),
        # ('Local Filter (R)', local_thres_running),
        # ('Otsu Filter', otsu_thres),
        # ## ('Adaptive Filter', adaptive),
        # ('Kittler', kittler),
        # ('yen', yen),
        # ('iso_data', isodata),
        # ('triangle', triangle),
        # ('minimum', minimum),
        # ('Canny', canny_filter),
        # ('CFAR', cfar),
        # ('sigma_clip', sigma_clipping),
        ('sigma_clip_map', sigma_clipping_map),
    ]

    for s in snr:
        metrics[s] = {}
        metrics_tmp_df = pd.DataFrame()
        print "\nEvaluating filters with tracks at SNR {:0.2f}W".format(s)
        # Create image from real data
        test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=512, nsamples=256)

        # Remove channels with RFI
        test_img = rfi_filter(test_img)

        # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

        # Estimate the noise
        noise_mean = np.mean(test_img)

        # Generate a number of tracks
        tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape, TRACK_THICKNESS)

        # Add tracks to the true data
        true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, noise_mean, s)

        # Add tracks to the simulated data
        test_img = add_tracks(test_img, tracks, noise_mean, s)

        visualise_image(test_img, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks, False)

        # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

        # Filter the data in generate metrics
        for f_name, filter_func in filters:
            data = test_img.copy()

            # split data in chunks
            start = time.time()
            # mask, threshold = chunked_filtering(data, filter_func)
            mask, threshold = filter_func(data)
            timing = time.time() - start

            # Visualise filter output
            metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)
            visualise_filter(test_img, mask, tracks, f_name, s, threshold, visualise=True)

            # Filter post-processor
            # mask, timing = morph_opening(data, test_img, mask)
            start = time.time()
            mask, _ = hit_and_miss(data, test_img, mask)
            timing = time.time() - start
            f_name += '_pp'
            metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)
            visualise_filter(test_img, mask, tracks, f_name, s, threshold=None)

        metrics_tmp_df = metrics_tmp_df.from_dict(metrics[s], orient='index')
        metrics_df = metrics_df.append(metrics_tmp_df)

    print metrics_df.sort_values(by=['score'], ascending=False)
