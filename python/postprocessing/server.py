import logging as log
import config.log as log_config

from controllers.OrbitDeterminationController import OrbitDeterminationController
from flask import Flask, render_template, Response
import json
import operator

app = Flask(__name__, template_folder='views')


@app.route("/birales/pointings/<observation>/<data_set>")
def view_pointings(observation='medicina_07_03_2016', data_set='1358'):
    return render_template('pointings.html', observation=observation, data_set=data_set)


@app.route("/birales/candidates/<observation>/<data_set>/images")
def view_candidates_in_beam_data_images(observation='medicina_07_03_2016', data_set='1358'):
    return render_template('waterfall_all.html', observation=observation, data_set=data_set, beams=range(0, 32))


@app.route("/birales/candidates/<observation>/<data_set>")
def view_candidates_in_beam_data_ajax(observation='medicina_07_03_2016', data_set='1358'):
    return render_template('waterfall_all_ajax.html', observation=observation, data_set=data_set, beams=range(0, 32))


@app.route("/birales/candidates/<observation>/<data_set>/<beam_id>")
def view_orbit_determination(observation='medicina_07_03_2016', data_set='1358', beam_id=15):
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
                           beam_id=int(beam_id))

@app.route("/birales/api/pointings/<observation>/<data_set>")
def api_pointings(observation='medicina_07_03_2016', data_set='1358'):
    od = OrbitDeterminationController()
    ref_pointing, pointings = od.get_pointings(observation, data_set)

    pointings = {
        'reference': ref_pointing,
        'beams': pointings,
    }

    return Response(json.dumps(pointings), mimetype='application/json')


@app.route("/birales/api/beam_data/candidates/<observation>/<data_set>/<beam_id>")
def api_candidates_in_beam_data(observation='medicina_07_03_2016', data_set='mock_1358', beam_id=15):
    od = OrbitDeterminationController()
    response = od.get_candidates_image(observation, data_set, beam_id)

    return response


@app.route("/birales/api/candidates/table/<observation>/<data_set>/<beam_id>")
def api_candidates_table(observation, data_set, beam_id):
    od = OrbitDeterminationController()
    candidates = od.get_candidates(observation, data_set, beam_id)
    processed_candidates = od.filter_candidate(candidates, min_time=1, min_frequency=409.995, max_frequency=410.005)

    ref_pointing, pointings = od.get_pointings(observation, data_set)

    return render_template('candidate_table.html', candidates=processed_candidates, observation=observation,
                           data_set=data_set,
                           beam_id=int(beam_id),
                           ref_pointing=ref_pointing,
                           pointings=pointings)

@app.route("/birales/api/candidates/<observation>/<data_set>")
def api_candidates_data(observation='medicina_07_03_2016', data_set='mock_1358'):
    od = OrbitDeterminationController()

    all_candidates = []
    beam_firing_time = {}
    for beam_id in range(0, 32):
        candidates = od.get_candidates(observation, data_set, beam_id)
        processed_candidates = od.filter_candidate(candidates, min_time=1, min_frequency=409.995, max_frequency=410.005)

        if processed_candidates:
            beam_firing_time[beam_id] = od.get_candidates_min_time(processed_candidates)
            all_candidates.append(processed_candidates)

    beam_firing_time = sorted(beam_firing_time.items(), key=operator.itemgetter(1))

    od = OrbitDeterminationController()
    ref_pointing, pointings = od.get_pointings(observation, data_set)

    pointings = {
        'reference': ref_pointing,
        'beams': pointings,
    }

    response = {
        'beam_firing_order': beam_firing_time,
        'candidates': all_candidates,
        'pointings': pointings
    }

    return Response(json.dumps(response), mimetype='application/json')


@app.route("/birales/api/beam_data/image/<observation>/<data_set>/<beam_id>")
def api_beam_data_image(observation='medicina_07_03_2016', data_set='mock_1358', beam_id=15):
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

if __name__ == "__main__":
    log.basicConfig(format=log_config.FORMAT, level=log.DEBUG)
    log.info('Server is running on localhost at http://localhost:5000/birales/<observation>/<data set>/<beam id>')
    app.run(debug=True, host='0.0.0.0')
