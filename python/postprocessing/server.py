import logging as log
import config.log as log_config

from controllers.OrbitDeterminationController import OrbitDeterminationController
from flask import Flask, render_template, Response
import json

app = Flask(__name__, template_folder='views')


@app.route("/birales/api/pointings/<observation>/<data_set>")
def api_pointings(observation='medicina_07_03_2016', data_set='1358'):
    od = OrbitDeterminationController()
    ref_pointing, pointings = od.get_pointings(observation, data_set)

    pointings = {
        'reference': ref_pointing,
        'beams': pointings,
    }

    return Response(json.dumps(pointings), mimetype='application/json')


@app.route("/birales/pointings/<observation>/<data_set>")
def pointings(observation='medicina_07_03_2016', data_set='1358'):
    return render_template('pointings.html', observation=observation, data_set=data_set)


@app.route("/birales/all_waterfalls/<observation>/<data_set>")
def w_orbit_determination(observation='medicina_07_03_2016', data_set='1358'):
    return render_template('waterfall_all.html', observation=observation, data_set=data_set, beams=range(0, 32))


@app.route("/birales/all_waterfalls2/<observation>/<data_set>")
def w_orbit_determination2(observation='medicina_07_03_2016', data_set='1358'):
    return render_template('waterfall_all_ajax.html', observation=observation, data_set=data_set, beams=range(0, 32))


@app.route("/birales/all_waterfalls/<observation>/<data_set>/<beam_id>/candidates")
def w_beam_data(observation='medicina_07_03_2016', data_set='mock_1358', beam_id=15):
    od = OrbitDeterminationController()
    response = od.get_candidates_image(observation, data_set, beam_id)

    return response


@app.route("/birales/rest/<observation>/<data_set>/candidates")
def get_candidates_time_channel(observation='medicina_07_03_2016', data_set='mock_1358'):
    od = OrbitDeterminationController()

    all_candidates = []
    for beam_id in range(0, 32):
        candidates = od.get_candidates(observation, data_set, beam_id)

        all_candidates.append(list(candidates))

    return Response(json.dumps(list(all_candidates)), mimetype='application/json')


@app.route("/birales/images/<observation>/<data_set>/<beam_id>")
def beam_data(observation='medicina_07_03_2016', data_set='mock_1358', beam_id=15):
    """
    Returns the beam_data for a particular observation > data set > beam

    :param observation: Observation name
    :param data_set: Data set name
    :param beam_id: Beam id
    :return:
    """
    od = OrbitDeterminationController()
    response = od.get_beam_data(observation, data_set, beam_id)

    return response


@app.route("/birales/<observation>/<data_set>/<beam_id>")
def orbit_determination(observation='medicina_07_03_2016', data_set='1358', beam_id=15):
    """
    Renders an html page with the candidates detected and beam data for a particular observation > data set > beam

    :param observation:
    :param data_set:
    :param beam_id:
    :return:
    """
    od = OrbitDeterminationController()
    candidates = od.get_candidates(observation, data_set, beam_id)
    return render_template('orbit_determination.html', candidates=list(candidates), observation=observation,
                           data_set=data_set,
                           beam_id=beam_id)


if __name__ == "__main__":
    log.basicConfig(format=log_config.FORMAT, level=log.DEBUG)
    log.info('Server is running on localhost at http://localhost:5000/birales/<observation>/<data set>/<beam id>')
    app.run(debug=True, host='0.0.0.0')
