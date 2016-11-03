var MultiBeam = function (observation, data_set) {
    var self = this;
    this.observation = observation;
    this.data_set = data_set;
    this.host = "http://127.0.0.1:5000";
    this.beam_config = {};

    this._plot_beam_firing_order = function (beam_firing_order) {
        var template_url = 'views/beam_firing_order.mustache';
        var selector = '#beam-firing-order';
        $.get(template_url, function (template) {
            $(selector).append(
                Mustache.render(template, {
                    order: beam_firing_order
                })
            );
        });
    };

    this._plot_beam_candidates = function (beam_candidates) {
        var title = 'Detected Beam Candidates';
        var x_label = 'Channel (MHz)';
        var y_label = 'Time (s)';
        var selector = 'beam-candidates-plot';

        var _get = function (object, key) {
            return $.map(object, function (a) {
                return a[key];
            });
        };

        var traces = [];

        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: _get(beam_candidate['detections'], 'frequency'),
                y: _get(beam_candidate['detections'], 'time'),
                mode: 'markers',
                name: 'beam ' + beam_candidate.beam_id + ' candidate ' + beam_candidate.name,
                marker: {
                    // size: _get(beam_candidate['detections'], 'snr'),
                    // sizeref: 0.1,
                }
            };
            traces.push(beam_candidates_trace);
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
    };

    this._plot_orbit_determination_data_table = function (beam_candidates) {
        var template_url = 'views/beam_candidate_table.mustache';
        var selector = '#orbit-determination-data-table';
        $.get(template_url, function (template) {
            $.each(beam_candidates, function (candidate_number, beam_candidate) {
                $(selector).append(
                    Mustache.render(template, {
                        detections: beam_candidate.detections,
                        beam_id: beam_candidate.beam_id,
                        candidate_id: beam_candidate.name,
                        beam_ra: self.beam_config[beam_candidate.beam_id].beam_ra,
                        beam_dec: self.beam_config[beam_candidate.beam_id].beam_dec
                    })
                );
            })
        });
    };

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

                    self.beam_config[beam_id] = {
                        'beam_ra': ra,
                        'beam_dec': dec
                    };

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

    this.plot_beam_candidates = function () {
        var data_url = self.host + "/monitoring/" + self.observation + "/" + self.data_set + "/multi_beam/beam_candidates";

        $.ajax({
            url: data_url,
            success: function (beam_data) {
                var beam_candidates = beam_data['candidates'];
                var beam_firing_order = beam_data['order'];

                // Plot the beam candidates
                self._plot_beam_candidates(beam_candidates);

                // Build the beam candidates data table
                self._plot_orbit_determination_data_table(beam_candidates);

                // Plot the beam firing order
                self._plot_beam_firing_order(beam_firing_order);
            }
        });
    };

    this.plot_filtered_beam_data = function (selector) {
        var data_url = self.host + "/monitoring/" + self.observation + "/" + self.data_set + "/beam/2/beam_detections";

        $.ajax({
            url: data_url,
            success: function (beams_data) {
                var title = 'Detections in Filtered Beams';
                var x_label = 'Channel (MHz)';
                var y_label = 'Time sample';
                var traces = [];
                $.each(beams_data, function (j, beam_data) {
                    var beam_candidates_trace = {
                        x: beam_data['channel'],
                        y: beam_data['time'],
                        mode: 'markers',
                        name: 'beam ' + beam_data['beam_id'],
                        marker: {
                            color: beam_data['snr']
                        }
                    };
                    traces.push(beam_candidates_trace);
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

    }
};
