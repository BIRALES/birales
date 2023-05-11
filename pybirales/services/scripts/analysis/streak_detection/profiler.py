from time import time
from typing import List, Tuple, Dict

import pandas as pd


class Profiler:
    """Usage -

    1)For more detailed breakdown per frame:

    add_timing()
    output()
    """

    timings: List[Tuple[str, float]] = []
    last_time: float = None

    def start(self):
        self.last_time = time()

    def reset(self):
        self.start()

    def add_timing(self, description: str):
        t = time()

        self.timings.append((description, t - self.last_time))

        self.last_time = t

    def show(self):
        pr = 262144 / 78125 * 5

        df = pd.DataFrame(self.timings, columns=['Name', 'Time'])
        df['% Real time'] = round(df['Time'] / pr * 100., 1)
        df['Time (%)'] = round(df['Time'] / df['Time'].sum() * 100., 1)
        df['Scaled x32'] = df['Time'] * 32
        df['Scaled x64'] = df['Time'] * 64
        df['Scaled x128'] = df['Time'] * 128

        print(df)

    def output(self) -> Dict:
        return {timing[0]: (timing[1]) for timing in self.timings}
