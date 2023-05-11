import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    csv_file = '/storage/data/birales/2022_02_23/CasA/cas_2022-02-23 12:04:02.100_offline_beamformed_data_stefcal_1_0.csv'

    df = pd.read_csv(csv_file)

    df[df['beam'] == 6].plot(y='power_db')
    df[df['beam'] == 15].plot(y='power_db')
    df[df['beam'] == 24].plot(y='power_db')
    df[df['beam'] == 30].plot(y='power_db')

    plt.show()
