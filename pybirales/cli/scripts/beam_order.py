import pandas as pd
import numpy as np
import datetime
import seaborn as sns
from pandas.io.json import json_normalize
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pybirales.modules.detection.repository import BeamCandidateRepository, ConfigurationRepository
from sklearn import linear_model


sns.set(context="notebook", style="white", color_codes=True, rc={'axes.axisbelow': False})
beam_candidates_repo = BeamCandidateRepository()
config_repo = ConfigurationRepository()
pd.set_option('display.width', 1200)
pd.set_option('display.max_rows', 500)

BEAM_ID = None
MAX_TIME = '2017-06-30 13:50:20'
MIN_TIME = '2017-06-30 13:49:20'


def get_data(beam_id, min_time, max_time):
    # Parse input data
    min_time = datetime.datetime.strptime(min_time, "%Y-%m-%d %H:%M:%S")
    max_time = datetime.datetime.strptime(max_time, "%Y-%m-%d %H:%M:%S")

    # Get beam candidates from database
    detected_beam_candidates = beam_candidates_repo.get(beam_id, None, None, max_time, min_time)

    # flatten results
    raw = json_normalize(detected_beam_candidates)[
        ['data.snr', 'data.channel', 'data.time', 'beam_id', 'noise', 'configuration_id']]

    # ensure that only 1 record is returned per beam
    return raw.groupby('beam_id').first().reset_index()


def process(raw):
    df = pd.DataFrame()
    df['time'] = pd.Series(list(set(chain.from_iterable(raw['data.time']))), dtype='datetime64[ns]')

    df = df.set_index('time')
    df = df.sort_index()

    for i in range(0, len(raw['beam_id'])):
        beam_id = str(raw['beam_id'][i])
        key2 = 'C' + beam_id
        key3 = 'S' + beam_id
        temp = pd.DataFrame({'time': raw['data.time'][i],
                             key3: raw['data.snr'][i],
                             key2: raw['data.channel'][i]
                             })
        temp = temp.set_index('time')
        df = df.join(temp, how='left')

    f = [col for col in list(df) if col.startswith('S')]
    df['max_snr'] = df.loc[:, f].max(axis=1)
    df['max_idx'] = df.loc[:, f].idxmax(axis=1)

    df['consecutive'] = df['max_idx'].groupby((df['max_idx'] != df['max_idx'].shift()).cumsum()).transform('size')
    df['dt'] = (df.index.to_series() - df.index.to_series().shift()).fillna(0).dt.microseconds
    mask = df['consecutive'] > 1
    return df[mask]


def visualise(df, pointings):
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title('SNR Profile (max peaks only)')
    ax1.set_ylabel('SNR (Db)')
    ax1.set_xlabel('Time')

    ax2.set_title('Illuminated Beams')
    ax2.set_ylabel('HourAngle')
    ax2.set_xlabel('deltaDEC')
    ax2.set_xlim([-2.5, 4.0])
    ax2.set_ylim([-2.5, 2.5])
    i_order = ''
    p_df = pd.DataFrame()
    for beam_id in df['max_idx'].unique().tolist():
        m = df['max_idx'] == beam_id
        i_order += beam_id[1:] + ', '
        ax1.plot(df.index.to_series()[m], df['max_snr'][m], marker='o', label='beam ' + str(beam_id[1:]))

        a = pd.DataFrame([[pointings[int(beam_id[1:])][0], pointings[int(beam_id[1:])][1]]
                      for _ in range(len(df['max_snr'][m]))])

        p_df = pd.concat([p_df, a])

        ax2.annotate(' ' + str(beam_id[1:]), (a[0][0], a[1][0]))
        ax2.plot(a[0], a[1], marker='o', label='beam ' + str(beam_id[1:]), ms=10)

    ax1.add_artist(AnchoredText("Illumination order: " + i_order[:-2], loc=2))
    print('Illumination order: {}'.format(i_order[:-2]))

    regression = linear_model.LinearRegression()
    x = p_df[0].values.reshape(len(p_df), 1)
    y = p_df[1].values.reshape(len(p_df), 1)
    regression.fit(x, y)

    plt.legend(bbox_to_anchor=(-1.2,-0.05), loc='upper left', ncol=14)

    plt.show()


raw_df = get_data(beam_id=None, min_time=MIN_TIME, max_time=MAX_TIME)
processed_df = process(raw_df)

beam_pointings = [[-3.2, -0.5], [-3.2, 0.5],
                  [-1.6, -2], [-1.6, -1.5], [-1.6, -1], [-1.6, -0.5], [-1.6, 0], [-1.6, 0.5], [-1.6, 1], [-1.6, 1.5],
                  [-1.6, 2],
                  [0, -2], [0, -1.5], [0, -1], [0, -0.5], [0, 0], [0, 0.5], [0, 1], [0, 1.5], [0, 2],
                  [1.6, -2], [1.6, -1.5], [1.6, -1], [1.6, -0.5], [1.6, 0], [1.6, 0.5], [1.6, 1], [1.6, 1.5], [1.6, 2],
                  [3.2, -0.5], [3.2, 0], [3.2, 0.5]]

visualise(processed_df, beam_pointings)
