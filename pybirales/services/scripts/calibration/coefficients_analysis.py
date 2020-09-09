import numpy as np
from matplotlib import pyplot as plt
from mongoengine import connect

from pybirales.repository.models import Observation

if __name__ == '__main__':
    db_connection = connect(
        db='birales',
        username='birales_rw',
        password='rw_Sept03',
        port=27017,
        host='192.167.189.74')

    max_obs = 3
    observations = Observation.objects(class_check=False, type='calibration', status='finished') \
        .order_by('-date_time_start').limit(max_obs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax1.set_title("Amplitude")
    ax2.set_title("Phase")
    markers = ['r.', 'g.', 'b.']
    for i, observation in enumerate(observations):
        antenna_id = range(0, 32)
        real = np.array(observation.real)
        imag = np.array(observation.imag)

        complex_gain = real + imag * 1j

        amplitude = np.absolute(complex_gain)
        phase = np.angle(complex_gain, deg=True)

        c = markers[i]
        obs_name = observation.created_at.strftime("%d/%m/%Y, %H:%M:%S")
        ax1.plot(antenna_id, amplitude, c, markersize=12, ls='',
                 markeredgewidth=1.0,
                 label=obs_name)

        ax2.plot(antenna_id, phase, c, markersize=12, ls='',
                 markeredgewidth=1.0,
                 label=obs_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
