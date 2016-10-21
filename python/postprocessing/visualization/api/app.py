from flask import Flask
from flask_restful import Api
from resources.beam import FilteredBeam, RawBeam
from resources.multi_beam import FilteredMultiBeam, RawMultiBeam
from resources.candidates import BeamCandidate, SpaceDebrisCandidate

app = Flask(__name__)
api = Api(app)

# Beam routes
api.add_resource(RawBeam,
                 '/monitoring/<string:observation>/<string:data_set>/beam/<int:beam_id>/raw/<string:plot_type>')
api.add_resource(FilteredBeam,
                 '/monitoring/<string:observation>/<string:data_set>/beam/<int:beam_id>/filtered/<string:plot_type>')

# Multi-beam routes
api.add_resource(RawMultiBeam, '/monitoring/<string:observation>/<string:data_set>/multi_beam/raw/<string:plot_type>')
api.add_resource(FilteredMultiBeam,
                 '/monitoring/<string:observation>/<string:data_set>/multi_beam/filtered/<string:plot_type>')

# Candidates routes
api.add_resource(BeamCandidate, '/monitoring/<string:observation>/<string:data_set>/beam/<int:beam_id>/candidates')
api.add_resource(SpaceDebrisCandidate,
                 '/monitoring/<string:observation>/<string:data_set>/multi_beam/space_debris_candidates')

if __name__ == '__main__':
    app.run(debug=True)
