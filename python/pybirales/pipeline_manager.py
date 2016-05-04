import time


class PipelineManager(object):
    """ Class to manage the pipeline """

    def __init__(self):
        """ Class constructor """
        self._modules = []
        self._module_names = []

    def add_module(self, name, module):
        """ Add a new module instance to the pipeline
        :param name: Name of the module instance
        :param module: Module instance
        """
        self._module_names.append(name)
        self._modules.append(module)

    def start_pipeline(self):
        """ Start running pipeline """
        try:
            # Start all modules
            for module in self._modules:
                module.start()
        except Exception as e:
            # An error occurred, force stop all modules
            self.stop_pipeline()

    def stop_pipeline(self):
        """ Stop pipeline (one at a time) """

        # Loop over all modules
        for module in self._modules:
            # Stop module
            module.stop()

            # Wait for module to stop
            while not module.is_stopped:
                time.sleep(0.5)

        # All done