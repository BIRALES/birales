import numpy as np


class MSReader:

    def __init__(self):

        self.UVW = None
        self.antenna1 = None
        self.antenna2 = None

    def read(self, UVW, antenna1, antenna2):

        self.UVW = UVW
        self.antenna1 = antenna1
        self.antenna2 = antenna2
