import os
import random

import numpy as np
from astropy.io import fits

from filters import rfi_filter

DOPPLER_RANGE = [2392, 6502]


def generate_line(x1, y1, x2, y2, limits=None):
    """Brensenham line algorithm"""

    steep = 0
    coords = []
    dx = abs(x2 - x1)
    if (x2 - x1) > 0:
        sx = 1
    else:
        sx = -1
    dy = abs(y2 - y1)
    if (y2 - y1) > 0:
        sy = 1
    else:
        sy = -1

    if dy > dx:
        steep = 1
        x1, y1 = y1, x1
        dx, dy = dy, dx
        sx, sy = sy, sx
    d = (2 * dy) - dx
    for i in range(0, dx):
        if steep:
            coords.append((y1, x1))
        else:
            coords.append((x1, y1))
        while d >= 0:
            y1 = y1 + sy
            d = d - (2 * dx)
        x1 = x1 + sx
        d = d + (2 * dy)
    coords.append((x2, y2))

    coords = np.array(coords)

    coords[:, [0, 1]] = coords[:, [1, 0]]

    return coords


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


def generate_test_image(directory, nof_samples):
    dir_list = [f for f in os.listdir(directory) if f.endswith('.fits')]  # get all fits files except the first one
    base_file = random.choice(dir_list)

    fits_file = fits.open(os.path.join(directory, base_file))

    start = np.random.randint(low=0, high=np.shape(fits_file[0].data[0])[1] - nof_samples + 1)

    test_img = fits_file[0].data[0, DOPPLER_RANGE[0]:DOPPLER_RANGE[1], start: start + nof_samples]

    print('Test Image of size {} was generated from Raw File {}. Noise estimate at {:0.3f}W'. \
          format(test_img.shape, base_file, np.mean(test_img)))

    # Remove channels with RFI
    test_img = rfi_filter(test_img)

    return test_img


def get_test_tracks0(n_tracks, gradient, track_length, image_shape, thickness):
    tracks = []
    dt = 0.10
    ds = 9.54 * 1e6
    n_tracks = np.random.randint(1, n_tracks)
    for i in range(0, n_tracks):
        m = np.random.uniform(gradient[0], gradient[1])
        tl = np.random.uniform(track_length[0], track_length[1])

        start = np.random.randint(low=0, high=image_shape[1] + 1 - tl)

        end = np.amin([start + tl, image_shape[1]])

        x = np.arange(start, end)
        c = np.random.randint(low=0, high=4110 - m * min(x))
        x, y = create_track(x, m, c, image_shape, thickness)

        print("Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) to ({},{})".format(m, c, (
                max(x) - start), start, max(y), end, min(y)))
        # beams = range(nof_beams)
        # reps = np.ceil(len(x) / float(len(beams)))
        #
        # b = np.repeat(beams, reps)[: len(x)]

        tracks.append(np.array([x, y]))

    print('Created {} tracks'.format(n_tracks))

    return tracks


def y(m, x, c):
    return m * x + c


def get_test_tracks11(n_tracks, gradient, track_length, image_shape, thickness):
    tracks = []
    dt = 0.09375
    ds = 9.5367431640625
    tx_idx = 4457

    tx_idx = 2065

    for i in range(0, 1):
        # slope = -61  # Hz / s
        # doppler = 11045  # Hz

        slope = np.random.uniform(-57, -291.47)  # Hz / s
        doppler = np.random.uniform(-13688, 13507)  # Hz

        slope_ = (slope / ds) / (1 / dt)
        doppler_ = tx_idx + doppler / ds

        tl = np.random.uniform(track_length[0], track_length[1])
        start = np.random.randint(low=0, high=image_shape[1] + 1 - tl)
        end = np.amin([start + tl, image_shape[1]])

        mid = int((end - start) / 2)

        c = doppler_ - slope_ * mid

        x = np.arange(start, end)
        x, y = create_track(x, slope_, c, image_shape, thickness)

        print("Created track of length {:0.2f} seconds and doppler={:0.2f} Hz and slope={:0.2f} Hz/s\n"
              "This translates to {} pixels, y={:0.2f} and slope={:0.2f}".format(len(x) * dt, doppler, slope, len(x),
                                                                                 doppler_, slope_))

        tracks.append(np.array([x, y]))

    print('Created {} tracks'.format(n_tracks))

    return tracks


def generate_track(slope, doppler, ds, dt, tx_idx, track_length, image_shape, thickness):
    slope_ = (slope / ds) / (1 / dt)
    doppler_ = tx_idx + doppler / ds

    tl = np.random.uniform(track_length[0], track_length[1])
    start = np.random.randint(low=0, high=image_shape[1] + 1 - tl)
    end = np.amin([start + tl, image_shape[1]])

    mid = int((end - start) / 2)

    c = doppler_ - slope_ * mid

    x = np.arange(start, end)
    x, y = create_track(x, slope_, c, image_shape, thickness)

    print("Created track of length {:0.2f} seconds and doppler={:0.2f} Hz and slope={:0.2f} Hz/s\n"
          "This translates to {} pixels, y={:0.2f} and slope={:0.2f}".format(len(x) * dt, doppler, slope, len(x),
                                                                             doppler_, slope_))

    return x, y


def get_test_tracks_crossing(n_tracks, gradient, track_length, image_shape, thickness):
    tracks = []
    dt = 0.09375
    ds = 9.5367431640625
    tx_idx = 4457

    tx_idx = 2065
    n_tracks = np.random.randint(1, n_tracks)

    targets = [(-280, 520), (-100, 400)]
    track_length = np.array([32, 50])
    for t in targets:
        slope = t[0]
        doppler = t[1]

        x, y = generate_track(slope, doppler, ds, dt, tx_idx, track_length, image_shape, thickness)

        tracks.append(np.array([x, y]))

    print('Created {} tracks'.format(n_tracks))

    return tracks


def get_test_tracks(n_tracks, gradient, track_length, image_shape, thickness):
    tracks = []
    dt = 0.09375
    ds = 9.5367431640625
    tx_idx = 4457

    tx_idx = 2065
    n_tracks = np.random.randint(3, n_tracks)
    for i in range(0, n_tracks):
        slope = np.random.uniform(-57, -291.47)  # Hz / s
        doppler = np.random.uniform(-13688, 13507)  # Hz

        # slope = -280.567  # Hz / s
        # doppler = 490.546  # Hz

        x, y = generate_track(slope, doppler, ds, dt, tx_idx, track_length, image_shape, thickness)

        tracks.append(np.array([x, y]))

    print('Created {} tracks'.format(n_tracks))

    return tracks

    n_tracks = np.random.randint(1, n_tracks)
    for i in range(0, n_tracks):
        m = np.random.uniform(gradient[0], gradient[1])
        tl = np.random.uniform(track_length[0], track_length[1])

        start = np.random.randint(low=0, high=image_shape[1] + 1 - tl)

        end = np.amin([start + tl, image_shape[1]])

        x = np.arange(start, end)
        c = np.random.randint(low=0, high=4110 - m * min(x))
        x, y = create_track(x, m, c, image_shape, thickness)

        print('Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) to ({},{})'.format(m, c, (
                max(x) - start), start, max(y), end, min(y)))
        # beams = range(nof_beams)
        # reps = np.ceil(len(x) / float(len(beams)))
        #
        # b = np.repeat(beams, reps)[: len(x)]

        tracks.append(np.array([x, y]))

    print('Created {} tracks'.format(n_tracks))

    return tracks


def get_test_tracks_old(n_tracks, gradient, track_length, image_shape, thickness):
    tracks = []
    n_tracks = np.random.randint(1, n_tracks)
    for i in range(0, n_tracks):
        m = np.random.uniform(gradient[0], gradient[1])
        tl = np.random.uniform(track_length[0], track_length[1])

        start = np.random.randint(low=0, high=image_shape[1] + 1 - tl)

        end = np.amin([start + tl, image_shape[1]])

        x = np.arange(start, end)
        c = np.random.randint(low=0, high=4110 - m * min(x))
        x, y = create_track(x, m, c, image_shape, thickness)

        print('Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) to ({},{})'.format(m, c, (
                max(x) - start), start, max(y), end, min(y)))
        # beams = range(nof_beams)
        # reps = np.ceil(len(x) / float(len(beams)))
        #
        # b = np.repeat(beams, reps)[: len(x)]

        tracks.append(np.array([x, y]))

    print('Created {} tracks'.format(n_tracks))

    return tracks


def add_tracks(image, tracks, noise_mean, snr):
    power = calculate_amplitude(noise_mean, snr=snr)

    for track in tracks:
        image[track[1], track[0]] += power
    return image
