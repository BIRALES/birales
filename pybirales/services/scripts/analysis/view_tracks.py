import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mongoengine import connect
import numpy as np
from pybirales.repository.models import SpaceDebrisTrack


def connect_db():
    return connect(
        db=os.environ['BIRALES__MONGO_DB'],
        username=os.environ['BIRALES__MONGO_USER'],
        password=os.environ['BIRALES__MONGO_PASSWORD'],
        port=27017,
        host='localhost')


def plot_snr_profile(df, title):
    _, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set(xlabel='Time Sample', ylabel='SNR (dB)')
    ax2.set(xlabel='Time Sample', ylabel='Beam')

    # df = df.sort_values('snr', ascending=False).drop_duplicates(subset=['time_sample']).sort_values(
    #     by=['time_sample'])

    for b in df['beam_id'].unique():
        data = df[df['beam_id'] == b].copy()
        deltas = data['time_sample'].diff()[1:]
        gaps = deltas[deltas > 20]
        data.loc[gaps.index, 'time_sample'] = np.nan

        ax1.plot(data['time_sample'], data['snr'], label=b, alpha=0.9, linewidth=1.0)

        ax2.plot(data['time_sample'], data['beam_id'], label=b, alpha=0.9, linewidth=7.0)

        groups = np.split(data, np.where(np.isnan(data.time_sample))[0])
        for g in groups:
            if not g['time_sample'].isnull().values.all() and len(g) > 1:
                ax2.annotate(b, xy=(np.mean(g['time_sample']), b), xycoords='data')
    ax2.set_yticks(np.arange(0, 32, step=2))
    ax2.grid()

    plt.savefig(f'{title}_beam_illumination.png', bbox_inches='tight')

    # illumination_sequence(df)

def plot_doppler_profile(df, title):
    _, ax = plt.subplots()
    ax.set(xlabel='Time Sample', ylabel='Channel Sample', title=title)
    for b in df['beam_id'].unique():
        ax = sns.scatterplot(y="channel_sample", x="time_sample", data=df[df['beam_id'] == b],
                             legend="auto", label=b)
    plt.savefig(f'{title}.png', bbox_inches='tight')


def save_detection_data(df, title):
    return df.to_csv(f'{title}.csv')


if __name__ == '__main__':
    connect_db()

    # UNCALIBRATED_TRACK_ID = "5fa3c7dbab7b60dc7c4d38f6"
    CALIBRATED_TRACK_ID = "624eaaab5af9deb89d1fb14d"
    CALIBRATED_TRACK_ID = "624eaaa85af9deb89d1fb144"
    CALIBRATED_TRACK_ID = "62502cf38f7863219645fc28"
    CALIBRATED_TRACK_ID = "62502e2b958db46044facab0"
    CALIBRATED_TRACK_ID = "625052d35043df9be8571b24"
    CALIBRATED_TRACK_ID = '625817c7c8f76a42b471b736' # NORAD39086 on 25/01
    # CALIBRATED_TRACK_ID = '' # NORAD39086 on 27/01

    # uncalibrated_track_df = pd.DataFrame(SpaceDebrisTrack.objects.get(_id=UNCALIBRATED_TRACK_ID).data)
    calibrated_track_df = pd.DataFrame(SpaceDebrisTrack.objects.get(_id=CALIBRATED_TRACK_ID).data)

    # plot_snr_profile(uncalibrated_track_df, title="SNR Profile (uncalibrated)")
    # plot_doppler_profile(uncalibrated_track_df, title="Doppler Curve (uncalibrated)")
    # save_detection_data(uncalibrated_track_df, "Detection data (uncalibrated)")

    plot_snr_profile(calibrated_track_df, title="SNR Profile")
    plot_doppler_profile(calibrated_track_df, title="Doppler Curve")
    # save_detection_data(calibrated_track_df, "Detection data (calibrated)")

    plt.show()
