import numpy as np
from astropy.io import fits


def calculate_amplitude(noise_avg, snr):
    """

    :param noise_std:
    :param snr:
    :return:
    """

    return 10 ** (snr / 10. - np.log10(noise_avg))




def create_track(x, gradient, intercept, img_shape, thickness):
    def thicken(x, y, t):
        x_new = np.array([x for _ in range(t)])
        y_new = np.array([y + i for i in range(t)])
        return x_new, y_new

    y = x * gradient + intercept

    x, y = thicken(x, y, thickness)

    y[y <= 0] = 0
    y[y >= img_shape[0]] = img_shape[0] - 1

    x[x <= 0] = 0
    x[x >= img_shape[1]] = img_shape[1] - 1

    return np.ravel(x.astype(int)), np.ravel(y.astype(int))


def create_test_img(filepath, nchans=192, nsamples=120):
    """

    :param filepath:
    :param scaling_factor: downscale factor
    :return:
    """
    fits_file = fits.open(filepath)
    size = np.shape(fits_file[0].data[0])

    # n_chans = size[0] - nchans
    # n_samples = size[1] - nsamples

    test_img = fits_file[0].data[0][:nchans, :nsamples]
    print 'Test Image of size {} was generated. Noise estimate at {:0.3f}W'.format(test_img.shape, np.mean(test_img))

    # return np.random.normal(0, 0.5, (nchans, nsamples))
    return test_img


def get_test_tracks(n_tracks, gradient, track_length, image_shape, thickness):
    tracks = []

    for i in range(0, n_tracks):
        m = np.random.uniform(gradient[0], gradient[1])
        start = np.random.randint(low=0, high=image_shape[1] / 3.)
        end = np.random.randint(low=start + track_length[0], high=start + track_length[1])
        end = np.amin([end, image_shape[1]])
        x = np.arange(start, end)
        c = np.random.randint(low=100, high=image_shape[0])
        x, y = create_track(x, m, c, image_shape, thickness)

        # print 'Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) to ({},{})'.format(m, c, (
        #         max(x) - start), start, max(y), end, min(y))

        tracks.append((x, y))
    print 'Created {} tracks'.format(n_tracks)
    return tracks


def add_tracks(image, tracks, noise_mean, snr):
    power = calculate_amplitude(noise_mean, snr=snr)

    for track in tracks:
        image[track[1], track[0]] += power
    return image
