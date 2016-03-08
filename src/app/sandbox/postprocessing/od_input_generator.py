import ephem
import numpy as np


class DateTimeHelper:

    def __init__(self):
        return

    @staticmethod
    def juldate2ephem(date):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
        return ephem.date(date - 2415020.)


    @staticmethod
    def ephem2juldate(date):
        """Convert ephem date (measured from noon, Dec. 31, 1899) to Julian date."""
        return float(date + 2415020.)


class OrbitDeterminationInputGenerator:

    noise_lvl = 0.2  # The standard deviation of the normal distribution noise

    def __init__(self):
        return

    def add_noise(self, noiseless_data):
        """
        Add noise to the passed numpy array
        :param noiseless_data:
        :return: noisy data
        """
        noise = abs(np.random.normal(0, self.noise_lvl, len(data)))
        return noiseless_data + noise


data = np.zeros(100)
od = OrbitDeterminationInputGenerator()
noisy_data = od.add_noise(data)

print noisy_data
