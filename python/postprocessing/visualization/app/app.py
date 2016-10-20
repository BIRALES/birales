from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


# Resources
# '<observation>/<data_set>/monitoring/beam/<beam id>/raw'
# '<observation>/<data_set>/monitoring/beam/<beam id>/filtered'
# '<observation>/<data_set>/monitoring/beam/<beam id>/candidates'
# '<observation>/<data_set>/monitoring/multi_beam/raw'
# '<observation>/<data_set>/monitoring/multi_beam/filtered'
# '<observation>/<data_set>/monitoring/multi_beam/space_debris_candidates'
# '<observation>/<data_set>/monitoring/multi_beam/pointings'


class Beam(Resource):
    def get(self):
        pass


class BeamCandidates(Resource):
    def get(self):
        pass


api.add_resource(BeamCandidates, '/monitoring/candidates/beam')

if __name__ == '__main__':
    app.run(debug=True)
