import logging as log
import time

from pybirales import settings
from pybirales.pipeline.base.definitions import BEST2PointingException, ROACHBackendException
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

            try:
                self._backend.start(program_fpga=True, equalize=True, calibrate=True)
            except RuntimeError:
                log.critical('Could not start the ROACH backend.')
                raise ROACHBackendException('Failed to start the ROACH backend.')

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

                self._instrument.connect()

            except BEST2PointingException:
                log.warning('BEST2 Server is not available.')
            else:
                if self._pointing_enabled:
                    log.info('Successfully connected to BEST antenna server and pointing is enabled')
                else:
                    log.info('Successfully connected to BEST and pointing is disabled')

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

    def get_declination(self):
        if self.is_enabled:
            try:
                dec = self._instrument.get_current_declination()
            except BEST2PointingException:
                log.warning('BEST2 Server is not available.')
            else:
                log.info("BEST-II current declination is: {:0.2f}".format(dec))
                return dec
        else:
            log.warning(
                'Could not establish the BEST antenna pointing. BEST-II pointing is disabled as'
                ' specified in the configuration.')

        return None

    def stop(self):
        if self.is_enabled and self._instrument:
            self._instrument.stop_best2_server()
