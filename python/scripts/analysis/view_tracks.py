import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mongoengine import connect

from pybirales.repository.models import SpaceDebrisTrack


def connect_db():
    return connect(
        db=os.environ['BIRALES__MONGO_DB'],
        username=os.environ['BIRALES__MONGO_USER'],
        password=os.environ['BIRALES__MONGO_PASSWORD'],
        port=27017,
        host='localhost')


def plot_snr_profile(df, title):
    _, ax = plt.subplots()
    ax.set(xlabel='Time Sample', ylabel='SNR (dB)', title=title)
    for b in df['beam_id'].unique():
        ax = sns.lineplot(y="snr", x="time_sample", data=df[df['beam_id'] == b], sort=True, ci=None, estimator=None,
                          legend="auto", label=b)

    plt.savefig(title+'.png', bbox_inches='tight')


def plot_doppler_profile(df, title):
    _, ax = plt.subplots()
    ax.set(xlabel='Time Sample', ylabel='Channel Sample', title=title)
    for b in df['beam_id'].unique():
        ax = sns.scatterplot(y="channel_sample", x="time_sample", data=df[df['beam_id'] == b],
                             legend="auto", label=b)
    plt.savefig(title+ '.png', bbox_inches='tight')


def save_detection_data(df, title):
    return df.to_csv(title+'.csv')


if __name__ == '__main__':
    connect_db()

    UNCALIBRATED_TRACK_ID = "5fa3c7dbab7b60dc7c4d38f6"
    CALIBRATED_TRACK_ID = "5fa3ca43819490cc509286a1"

    uncalibrated_track_df = pd.DataFrame(SpaceDebrisTrack.objects.get(_id=UNCALIBRATED_TRACK_ID).data)
    calibrated_track_df = pd.DataFrame(SpaceDebrisTrack.objects.get(_id=CALIBRATED_TRACK_ID).data)

    plot_snr_profile(uncalibrated_track_df, title="SNR Profile (uncalibrated)")
    plot_doppler_profile(uncalibrated_track_df, title="Doppler Curve (uncalibrated)")
    save_detection_data(uncalibrated_track_df, "Detection data (uncalibrated)")

    plot_snr_profile(calibrated_track_df, title="SNR Profile (calibrated)")
    plot_doppler_profile(calibrated_track_df, title="Doppler Curve (calibrated)")
    save_detection_data(calibrated_track_df, "Detection data (calibrated)")

    plt.show()
