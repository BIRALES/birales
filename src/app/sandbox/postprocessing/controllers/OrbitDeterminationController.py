import pymongo as mongo
import app.sandbox.postprocessing.config.database as DB
from app.sandbox.postprocessing.models.Observation import Observation
from matplotlib import pyplot as plt


class OrbitDeterminationController:
    def __init__(self):
        client = mongo.MongoClient(DB.HOST, DB.PORT)
        self.db = client['birales']

    def get_candidates(self, observation, data_set, beam_id):
        query = {
            "observation": observation.name,
            "data_set"   : data_set,
            "beam"       : str(beam_id),
        }

        return self.db.candidates.find(query)

    def get_beam_data(self, observation, data_set, beam_id):
        observation = Observation(observation, data_set, tx = 100)

        beam = observation.beams[beam_id]
        candidates = self.get_candidates(observation, data_set, beam_id)

        exit(0)



    @staticmethod
    def view_candidates(beam, candidates):
        fig = plt.figure(figsize = (8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %s" % beam.id)
        plt.imshow(beam.snr.transpose(), aspect = 'auto', interpolation = "none", origin = "lower",
                   extent = [beam.channels[0], beam.channels[len(beam.snr)], 0, beam.time[-1]])
        ax.set_xlabel("Channel (kHz)")
        ax.set_ylabel("Time (s)")

        for candidate in candidates:
            plt.plot(candidate['data']['frequency'], candidate['data']['time'], 'o')

        plt.show()
