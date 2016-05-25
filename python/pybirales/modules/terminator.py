from pybirales.base.processing_module import ProcessingModule
import time


class Terminator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # Call superclass initialiser
        super(Terminator, self).__init__(config, input_blob)

        self._prev_time = time.time()

        # Processing module name
        self.name = "Terminator"

    def generate_output_blob(self):
        """ Generate output data blob """
        return None

    def process(self, obs_info, input_data, output_data):
        # Calculate processed seconds
        print "Processed %d samples in %.2fs" % (obs_info['nsamp'], time.time() - self._prev_time)
        self._prev_time = time.time()
        return obs_info
