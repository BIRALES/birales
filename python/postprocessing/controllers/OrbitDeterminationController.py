import pymongo as mongo
import config.database as DB
import itertools
import StringIO

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.Observation import Observation
from matplotlib import pyplot as plt
from flask import make_response


class OrbitDeterminationController:
    def __init__(self):
        client = mongo.MongoClient(DB.HOST, DB.PORT)
        self.db = client['birales']

    def get_candidates(self, observation, data_set, beam_id):
        query = {
            "observation": observation,
            "data_set": data_set,
            "beam": int(beam_id),
        }
        return self.db.candidates.find(query)

    def get_beam_data(self, observation, data_set, beam_id):
        observation = Observation(observation, data_set)
        beam = observation.beams[int(beam_id)]
        candidates = self.get_candidates(observation.name, data_set, beam_id)
        return self.view_candidates(beam, candidates)

    @staticmethod
    def view_candidates(beam, candidates, show_beam_data=True):
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %s" % beam.id)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Time (s)")
        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w', 'y']
        color = itertools.cycle(colors)

        for i, candidate in enumerate(list(candidates)):
            c = next(color)
            ax.plot(candidate['data']['frequency'], candidate['data']['time'], 'o', color=c,
                    label="Candidate " + str(i + 1))

        if show_beam_data:
            ax.imshow(beam.snr.transpose(), aspect='auto', origin="lower",
                      extent=[beam.channels[0], beam.channels[-1], 0, beam.time[-1]])
        ax.ticklabel_format(useOffset=False)

        # plt.show()
        ax.legend()
        canvas = FigureCanvas(fig)
        output = StringIO.StringIO()
        canvas.print_png(output)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        return response

    def get_candidates_image(self, observation, data_set, beam_id):
        candidates = self.get_candidates(observation, data_set, beam_id)
        return self.view_candidates_only(candidates, beam_id)

    @staticmethod
    def view_candidates_only(candidates, beam_id):
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %s" % beam_id)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Time (s)")
        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w', 'y']
        color = itertools.cycle(colors)

        for i, candidate in enumerate(list(candidates)):
            c = next(color)
            ax.plot(candidate['data']['frequency'], candidate['data']['time'], 'o', color=c,
                    label="Candidate " + str(i + 1))
        ax.ticklabel_format(useOffset=False)

        # plt.show()
        ax.legend()
        canvas = FigureCanvas(fig)
        output = StringIO.StringIO()
        canvas.print_png(output)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        return response
