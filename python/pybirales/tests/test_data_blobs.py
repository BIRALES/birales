from pybirales.pipeline.base.blob import DataBlob
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.pipeline import PipelineManagerBuilder

nof_iterations = 1e3


class TestDataBlob(DataBlob):
    pass


class TestProducer(ProcessingModule):

    def __init__(self, config, input_blob=None):
        super().__init__(config, input_blob)
        self._running_counter = 0

    def generate_output_blob(self):
        return TestDataBlob([('values', 1)], datatype=int)

    def process(self, obs_info, input_data, output_data):
        # If the maximum counter is reached, trigger to stop pipeline
        if self._running_counter >= nof_iterations:
            obs_info['stop_pipeline_at'] = self._iter_count
            self.stop()
            return obs_info

        # Produce next value
        output_data[0] = 1
        self._running_counter += 1
        return obs_info

    @property
    def running_counter(self):
        return self._running_counter


class TestConsumer(ProcessingModule):

    def __init__(self, config, input_blob=None):
        super().__init__(config, input_blob)
        self._running_sum = 0

    def generate_output_blob(self):
        return TestDataBlob([('values', 1)], datatype=int)

    def process(self, obs_info, input_data, output_data):
        output_data[0] = input_data[0]
        self._running_sum += output_data[0]
        return obs_info

    @property
    def running_sum(self):
        return self._running_sum


class TestPipeline(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)
        self.manager.name = 'Test Pipeline'
        self._id = "test_pipeline_builder"

        self._consumers = []

    def build(self):
        producer = TestProducer(None)
        producer.set_name("Producer")
        consumer_a = TestConsumer(None, producer.output_blob)
        consumer_a.set_name("Consumer A")
        consumer_b = TestConsumer(None, producer.output_blob)
        consumer_b.set_name("Consumer B")
        consumer_c = TestConsumer(None, consumer_a.output_blob)
        consumer_c.set_name("Consumer C")
        consumer_d = TestConsumer(None, consumer_a.output_blob)
        consumer_d.set_name("Consumer D")
        consumer_e = TestConsumer(None, consumer_a.output_blob)
        consumer_e.set_name("Consumer E")
        consumer_f = TestConsumer(None, consumer_b.output_blob)
        consumer_f.set_name("Consumer F")

        self.manager.add_module('producer', producer)
        self.manager.add_module('consumer_a', consumer_a)
        self.manager.add_module('consumer_b', consumer_b)
        self.manager.add_module('consumer_c', consumer_c)
        self.manager.add_module('consumer_d', consumer_d)
        self.manager.add_module('consumer_e', consumer_e)
        self.manager.add_module('consumer_f', consumer_f)

        self._consumers = [consumer_a, consumer_b, consumer_c, consumer_d, consumer_e, consumer_f]

    def check_results(self):
        assert all([c.running_sum == nof_iterations for c in self._consumers])
        print("All tests passed")


if __name__ == '__main__':

    pipeline = TestPipeline()
    pipeline.build()

    pipeline.manager.start_pipeline(standalone=True)

    # Once pipeline is finished, determine whether all consumers consumed the same number of blobs
    pipeline.check_results()
