import logging as log
import time

from pybirales import settings
from pybirales.pipeline.base.definitions import BEST2PointingException
from pybirales.services.instrument.backend import Backend
from pybirales.services.instrument.best2 import BEST2


class BackendController():

    def __init__(self):
        """
        Initialisation of the backend system
        """

        self._online = not settings.manager.offline

        if self.is_enabled:
            log.info('Loading backend')
            self._backend = Backend.Instance()
            time.sleep(1)
            self._backend.start(program_fpga=True, equalize=True, calibrate=True)

            log.info('Backend loaded')
        else:
            log.info('Backend not initialised as the observation is in offline mode.')

    @property
    def is_enabled(self):
        if self._online:
            return True

        return False

    def stop(self):
        if self.is_enabled:
            self._backend.stop()

class InstrumentController():
    def __init__(self):
        self._online = not settings.manager.offline

        self._pointing_enabled = settings.instrument.enable_pointing

        self._instrument = None

        if self.is_enabled:
            try:
                self._instrument = BEST2.Instance()

                if self._pointing_enabled:
                    self._instrument.connect()

            except BEST2PointingException:
                log.warning('BEST2 Server is not available.')
            else:
                if self._pointing_enabled:
                    log.info('Successfully connected to BEST antenna server')
                else:
                    log.info('System is not connected to the BEST antenna server as specified in the configuration')

    @property
    def is_enabled(self):
        if self._online:
            return True

        return False

    def point(self, declination):
        """
        Point the BEST Antenna

        :param declination:
        :return:
        """
        if self.is_enabled and self._pointing_enabled:
            try:
                self._instrument.move_to_declination(declination)
            except BEST2PointingException:
                log.warning('Could not point the BEST2 Antenna to DEC: {:0.2f}.'.format(declination))
                raise
        else:
            log.warning('Could not point to %s. BEST-II pointing is disabled as specified in the configuration.',
                        declination)

    def stop(self):
        if self.is_enabled and self._instrument:
                self._instrument.stop_best2_server()