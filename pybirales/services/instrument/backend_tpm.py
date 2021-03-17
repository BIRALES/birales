import logging
import struct
import time
import os

import numpy as np

from pybirales.birales_config import BiralesConfig
from pybirales.utilities.singleton import Singleton
from pybirales.digital_backend import digital_backend
from pybirales import settings


@Singleton
class TPMBackend(object):
    def __init__(self):
        """ TPM Digital Backend """

        self._station = None

    def connect(self):
        """ Connect to station """
        # If station is not reachable set, return
        if self._station is None:
            return False

        try:
            self._station.connect()
        except Exception as e:
            logging.error("Could not connect to digital backend, check configuration and hardware: {}".format(e))
            return False

        return True

    def start(self, program=True, initialise=True, calibrate=False):
        """ Start the digital backend """

        # Load station configuration
        digital_backend.load_configuration_file(settings.digital_backend.configuration_file)

        # Update configuration to match programming and initializati
        station_config = digital_backend.Station(digital_backend.configuration).configuration

        # Load station configuration
        try:
            self._station = digital_backend.Station(station_config)
        except Exception as e:
            logging.error("Could not configure station, please check configuration file: {}".format(e))
            return False

        # If we need to program and/or initialise the station, or if the station is not properly
        # set up, set config and re-connect
        if program or initialise or not self._station.properly_formed_station:
            # Digital backend not configured properly, re-configure
            station_config['station']['program'] = program
            station_config['station']['initialise'] = initialise

            try:
                # Program and initialise station
                self._station.connect()

                # Equalize signals if required
                self._station.equalize_ada_gain(16)

            except Exception as e:
                logging.error("Could not configure digital backend, check configuration and hardware: {}".format(e))
            else:
                if not self._station.properly_formed_station:
                    logging.error("Could not configure digital backend, check configuration and hardware")
            finally:
                digital_backend.configuration['station']['program'] = False
                digital_backend.configuration['station']['initialise'] = False

        elif not self.connect():
                return False

    def stop(self):
        """ Stop methode, does nothing for TPM """
        pass


if __name__ == "__main__":
    BiralesConfig().load()
    station = TPMBackend.Instance()
    station.start()
