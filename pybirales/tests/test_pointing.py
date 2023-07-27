from pybirales import settings
from pybirales.birales_config import BiralesConfig
from pybirales.pipeline.modules.beamformer.pointing import Pointing

if __name__ == "__main__":
    config = BiralesConfig(['/home/lessju/Software/birales/pybirales/configuration/birales.ini'])

    # Override some settings
    settings.beamformer.calibrate_subarrays = False
    settings.beamformer.nof_subarrays = 4
    settings.beamformer.reference_declinations = [40] * settings.beamformer.nof_subarrays
    settings.beamformer.nof_beams_per_subarray = 64

    pointing = Pointing(settings.beamformer, 1, 128)
