from app.sandbox.postprocessing.controllers.OrbitDeterminationController import OrbitDeterminationController
import app.sandbox.postprocessing.config.log as log_config

import logging as log


log.basicConfig(format = log_config.FORMAT, level = log.DEBUG)


od = OrbitDeterminationController()
od.get_beam_data(observation = 'medicina_07_03_2016', data_set = 'mock_1358', beam_id = 15)