import logging as log
import config.log as log_config

from controllers.OrbitDeterminationController import OrbitDeterminationController
from flask import Flask, render_template

app = Flask(__name__, template_folder = 'views')


@app.route("/birales/images/<observation>/<data_set>/<beam_id>")
def beam_data(observation = 'medicina_07_03_2016', data_set = 'mock_1358', beam_id = 15):
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
    return render_template('orbit_determination.html', candidates = list(candidates), observation = observation, data_set = data_set,
                           beam_id = beam_id)

if __name__ == "__main__":
    log.basicConfig(format = log_config.FORMAT, level = log.DEBUG)
    log.info('Server is running on localhost at http://localhost:5000/birales/<observation>/<data set>/<beam id>')
    app.run(debug = True, host='0.0.0.0')
