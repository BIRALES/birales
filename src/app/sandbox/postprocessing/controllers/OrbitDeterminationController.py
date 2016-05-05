import pymongo as mongo
import app.sandbox.postprocessing.config.database as DB
from app.sandbox.postprocessing.models.Observation import Observation
from matplotlib import cm, pyplot as plt
from flask import make_response
import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import itertools


class OrbitDeterminationController:
    def __init__(self):
        client = mongo.MongoClient(DB.HOST, DB.PORT)
        self.db = client['birales']

    def get_candidates(self, observation, data_set, beam_id):
        print observation
        query = {
            "observation": observation,
            "data_set":    data_set,
            "beam":        str(beam_id),
        }

        return self.db.candidates.find(query)

    def get_beam_data(self, observation, data_set, beam_id):
        observation = Observation(observation, data_set, tx = 100)
        beam = observation.beams[int(beam_id)]
        candidates = self.get_candidates(observation.name, data_set, beam_id)
        return self.view_candidates(beam, candidates)

    @staticmethod
    def view_candidates(beam, candidates):
        fig = plt.figure(figsize = (8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %s" % beam.id)
        ax.imshow(beam.snr.transpose(), aspect = 'auto', interpolation = "none", origin = "lower",
                  extent = [beam.channels[0], beam.channels[len(beam.snr)], 0, beam.time[-1]])
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Time (s)")
        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w', 'y']
        color = itertools.cycle(colors)
        for i, candidate in enumerate(list(candidates)):
            c = next(color)
            ax.plot(candidate['data']['frequency'], candidate['data']['time'], 'o', color = c, label="Candidate " + str(i+1))

        # plt.show()
        ax.legend()
        canvas = FigureCanvas(fig)
        output = StringIO.StringIO()
        canvas.print_png(output)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        return response
