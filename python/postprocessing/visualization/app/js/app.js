$(document).ready(function () {
    var observation = "medicina_26_05_2016";
    var data_set = "norad_29499";

    var multi_beam = new MultiBeam(observation, data_set);
    multi_beam.plot_beam_configuration("beam-candidates-plot");
    multi_beam.plot_beam_candidates("beam-candidates-plot");

    // todo
    // multi_beam.plot_beam_firing_order();
    // multi_beam.plot_orbit_determination_data_table();
    // multi_beam.plot_filtered_beam_data();
});

var MultiBeam = function (observation, data_set) {
    var self = this;
    this.observation = observation;
    this.data_set = data_set;
    this.host = "http://127.0.0.1:5000";

    this.plot_beam_configuration = function (selector) {
        var data_url = self.host + "/monitoring/" + self.observation + "/" + self.data_set + "/multi_beam/configuration";
        var title = 'Multi-Beam';
        var x_label = 'ra (deg)';
        var y_label = 'dec (deg)';

        $.ajax({
            url: data_url,
            success: function (pointings) {
                var reference = pointings['config']['reference_pointing'];
                var beam_pointings = pointings['config']['pointings'];
                var shapes = [];
                var data = [{
                    x: [],
                    y: [],
                    text: [],
                    mode: 'text'
                }];

                $(beam_pointings).each(function (beam_id, pointing_data) {
                    var ra = pointing_data[0];
                    var dec = pointing_data[1];
                    var bw_ra = 1.75;
                    var bw_dec = 0.5;

                    var shape = {
                        type: 'circle',
                        xref: 'x',
                        yref: 'y',
                        x0: reference[0] + ra,
                        x1: reference[0] + ra + bw_ra,
                        y0: reference[1] + dec,
                        y1: reference[1] + dec + bw_dec
                    };

                    data[0].x.push(reference[0] + ra + (bw_ra * 0.5));
                    data[0].y.push(reference[1] + dec + (bw_dec * 0.5));
                    data[0].text.push(beam_id);

                    shapes.push(shape);
                });

                var layout = {
                    title: title,
                    xaxis: {
                        autorange: true,
                        title: x_label
                    },
                    yaxis: {
                        autorange: true,
                        title: y_label
                    },
                    width: 500,
                    height: 500,
                    shapes: shapes
                };

                Plotly.newPlot(selector, data, layout);
            }
        });
    };

    this.plot_beam_candidates = function (selector) {
        var data_url = self.host + "/monitoring/" + self.observation + "/" + self.data_set + "/multi_beam/beam_candidates";
        var title = 'Beam Candidates Detected';
        var x_label = 'Channel (MHz)';
        var y_label = 'Time sample';

        $.ajax({
            url: data_url,
            success: function (beam_data) {
                var beams = beam_data['beams'];
                var traces = [];
                $.each(beams, function (i, beam) {
                    $.each(beam, function (j, beam_candidate) {
                        var beam_candidates_trace = {
                            x: beam_candidate.data['frequency'],
                            y: beam_candidate.data['time'],
                            mode: 'markers',
                            type: 'scatter',
                            name: 'beam ' + beam_candidate.beam_id + ' candidate ' + j,
                            marker: {
                                size: beam_candidate.data['snr'] * 10.
                            }
                        };
                        traces.push(beam_candidates_trace)
                    });
                });

                var layout = {
                    title: title,
                    xaxis: {
                        title: x_label
                    },
                    yaxis: {
                        title: y_label
                    }
                };

                Plotly.newPlot(selector, traces, layout);
            }
        });
    };
};
