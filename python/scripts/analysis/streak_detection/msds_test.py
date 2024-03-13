import logging

from pybirales.pipeline.modules.detection.filter import triangle_filter, PepperNoiseFilter
from pybirales.pipeline.modules.detection.msds.msds import *
from pybirales.pipeline.modules.detection.msds.visualisation import *
from receiver import *


# global_process_pool = Pool(8)

def read_fits_file(filepath, doppler_range):
    doppler_start, doppler_end = doppler_range
    fits_file = fits.open(filepath)

    test_img = fits_file[0].data[0, doppler_start:doppler_end, :]

    return test_img


def rfi_filter(input_image):
    pnf = PepperNoiseFilter()

    output_image = triangle_filter(input_image, None)

    mask = pnf._remove_pepper_noise(output_image)
    output_image[mask] = 0

    return output_image


def msds(test_image, true_tracks, noise_est, debug):
    limits = get_limits(test_image, true_tracks)

    pub = True
    ext = '.pdf'

    # limits = (0, 70, 2000, 2160)   #s limits for crossing streaks
    # limits = (50,120, 1100, 1400)
    # limits = None

    ndx = pre_process_data(test_image)

    # Build quad/nd tree that spans all the data points
    k_tree = build_tree(ndx, leave_size=40, n_axis=2)

    visualise_filtered_data(ndx, true_tracks, '1_filtered_data' + ext, limits=limits, debug=debug, pub=pub)

    # Traverse the tree and identify valid linear streaks
    leaves = traverse(k_tree.tree, ndx, bbox=(0, test_image.shape[1], 0, test_image.shape[0]), min_length=2.,
                      noise_est=noise_est)

    positives = process_leaves(leaves)

    print("Processed {} leaves. Of which {} were positives.".format(len(leaves), len(positives)))

    visualise_tree_traversal(ndx, true_tracks, positives, leaves, '2_processed_leaves' + ext, limits=limits,
                             vis=True, pub=True)
    eps = estimate_leave_eps(positives)

    print('eps is:', eps)
    cluster_data = h_cluster_leaves(positives, distance_thold=eps)

    visualise_clusters(cluster_data, true_tracks, positives,
                       filename='3_clusters' + ext,
                       limits=limits,
                       debug=debug, pub=pub)
    # Filter invalid clusters
    tracks = validate_clusters(cluster_data)

    visualise_tracks(tracks, true_tracks, '4_tracks' + ext, limits=limits, debug=debug, pub=pub)

    visualise_tracks(tracks, true_tracks, '5_tracks' + ext, limits=None, debug=debug, pub=pub)

    return tracks


if __name__ == '__main__':
    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    str_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    filepath = "/home/denis/.birales/visualisation/fits/41420_test/Observation_2022-01-25T1609_filtered_0.fits"
    doppler_range = (1000, 1200)

    input_image = read_fits_file(filepath, doppler_range)
    # filtered_image = rfi_filter(input_image)

    msds(input_image, None, noise_est=1.239, debug=True)
