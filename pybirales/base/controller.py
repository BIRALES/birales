import logging as log
import time

from pybirales import settings
from pybirales.pipeline.base.definitions import TPMBackendException
from pybirales.services.instrument.backend_tpm import TPMBackend


class BackendController:

    def __init__(self):
        """
        Initialisation of the backend system
        """

        self._online = not settings.manager.offline

        if self.is_enabled:
            log.info('Loading backend')
            # self._backend = Backend.Instance()
            self._backend = TPMBackend.Instance()
            time.sleep(1)

            try:
                self._backend.start(program=False, initialise=False, calibrate=False)
            except RuntimeError:
                # log.critical('Could not start the ROACH backend.')
                # raise ROACHBackendException('Failed to start the ROACH backend.')
                log.critical('Could not start the TPM backend.')
                raise TPMBackendException('Failed to start the TPM backend.')

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
