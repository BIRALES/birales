from pybirales.modules.detection.repository import BeamCandidateRepository
import pandas as pd
import datetime
from pandas.io.json import json_normalize
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

sns.set(context="notebook", style="white", color_codes=True, rc={'axes.axisbelow': False})

beam_candidates_repo = BeamCandidateRepository()
pd.set_option('display.width', 1200)
pd.set_option('display.max_rows', 500)

BEAM_ID = None
MAX_TIME = '2017-06-30 15:40:00'
MIN_TIME = '2017-06-30 13:34:00'


def get_data(beam_id, min_time, max_time):
    # Parse input data
    min_time = datetime.datetime.strptime(min_time, "%Y-%m-%d %H:%M:%S")
    max_time = datetime.datetime.strptime(max_time, "%Y-%m-%d %H:%M:%S")

    # Get beam candidates from database
    detected_beam_candidates = beam_candidates_repo.get(beam_id, None, None, max_time, min_time)

    # flatten results
    raw = json_normalize(detected_beam_candidates)[['data.snr', 'data.channel', 'data.time', 'beam_id']]

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


def visualise(df):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('SNR Profile (max peaks only)')
    ax.set_ylabel('SNR (Db)')
    ax.set_xlabel('Time')
    i_order = ''
    for beam_id in df['max_idx'].unique().tolist():
        m = df['max_idx'] == beam_id
        i_order += beam_id[1:] + ', '
        ax.plot(df.index.to_series()[m], df['max_snr'][m], marker='o', label='beam ' + str(beam_id[1:]))

    ax.add_artist(AnchoredText("Illumination order: " + i_order, loc=2))

    plt.legend()
    plt.show()


raw_df = get_data(beam_id=None, min_time=MIN_TIME, max_time=MAX_TIME)
processed_df = process(raw_df)
visualise(processed_df)
