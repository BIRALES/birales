from flask import Flask
from flask_restful import Api
from resources.candidates import BeamCandidate, MultiBeamCandidate, MultiBeamDetections
from resources.data_set import DataSet
from flask_cors import CORS


def init_api_resources(api):

    pre_fix_route = '/monitoring/<string:observation>/<string:data_set>/'

    # Detections
    api.add_resource(MultiBeamDetections, pre_fix_route + 'beam/<int:beam_id>/beam_detections')

    # Candidates routes
    api.add_resource(MultiBeamCandidate, pre_fix_route + 'multi_beam/beam_candidates')

    # Data set routes
    api = api.add_resource(DataSet,  pre_fix_route + 'multi_beam/configuration')

    return api


def run_server(port=5000):
    app = Flask(__name__)
    CORS(app)
    api = Api(app)

    api = init_api_resources(api)
    app.run(debug=True, port=port)


if __name__ == '__main__':
    run_server()
