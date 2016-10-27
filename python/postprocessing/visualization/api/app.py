from flask import Flask
from flask_restful import Api
from resources.beam import FilteredBeam, RawBeam
from resources.candidates import BeamCandidate, SpaceDebrisCandidate
from resources.data_set import DataSet
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
api = Api(app)

# Beam routes
api.add_resource(RawBeam,
                 '/monitoring/<string:observation>/<string:data_set>/beam/<int:beam_id>/raw/<string:plot_type>')
api.add_resource(FilteredBeam,
                 '/monitoring/<string:observation>/<string:data_set>/beam/<int:beam_id>/filtered/<string:plot_type>')

# Candidates routes
api.add_resource(BeamCandidate, '/monitoring/<string:observation>/<string:data_set>/beam/<int:beam_id>/candidates')
api.add_resource(SpaceDebrisCandidate,
                 '/monitoring/<string:observation>/<string:data_set>/multi_beam/space_debris_candidates')

# Data sets routes
api.add_resource(DataSet, '/monitoring/<string:observation>/<string:data_set>/configuration')

if __name__ == '__main__':
    app.run(debug=True)
