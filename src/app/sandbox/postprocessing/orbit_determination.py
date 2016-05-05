from app.sandbox.postprocessing.controllers.OrbitDeterminationController import OrbitDeterminationController
from flask import Flask, render_template

app = Flask(__name__, template_folder = 'views')


@app.route("/birales/images/<observation>/<data_set>/<beam_id>")
def beam_data(observation = 'medicina_07_03_2016', data_set = 'mock_1358', beam_id = 15):
    od = OrbitDeterminationController()
    response = od.get_beam_data(observation, data_set, beam_id)

    return response


@app.route("/birales/<observation>/<data_set>/<beam_id>")
def orbit_determination(observation = 'medicina_07_03_2016', data_set = 'mock_1358', beam_id = 15):
    od = OrbitDeterminationController()
    candidates = od.get_candidates(observation, data_set, beam_id)

    return render_template('test.html', candidates = list(candidates), observation = observation, data_set = data_set,
                           beam_id = beam_id)


if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0')
