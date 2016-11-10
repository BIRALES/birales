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

    this._plot_beam_illumination_timeline = function (beam_candidates) {

        var title = 'Beam Illumination Time line';
        var x_label = 'Time (s)';
        var y_label = 'Beam';
        var selector = 'beam-illumination-time-line-plot';

        var _get = function (object, key) {
            return $.map(object, function (a) {
                return a[key];
            });
        };

        var traces = [];

        $.each(beam_candidates, function (j, beam_candidate) {
            x_data = _get(beam_candidate['detections'], 'time');
            y_data = [];

            $.each(x_data, function (i, e) {
                y_data.push(beam_candidate.beam_id)
            });
            var beam_candidates_trace = {
                x: x_data,
                y: y_data,
                mode: 'markers',
                name: 'Candidate ' + beam_candidate.name,
            };
            traces.push(beam_candidates_trace);
        });

        var layout = {
            title: title,
            xaxis: {
                title: x_label
            },
            yaxis: {
                dtick: 1.0,
                title: y_label
            }
        };

        Plotly.newPlot(selector, traces, layout);
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

    this.plot_data_set_data = function () {
        var data_url = self.host + "/monitoring/" + self.observation + "/" + self.data_set + "/about";

        $.ajax({
            url: data_url,
            success: function (data_set) {
                // Display the beam plot configuration
                self._plot_beam_configuration('multi-beam-configuration-plot', data_set);

                // Display the Data set information table
                self._display_data_set_info_table('data-set-info-table', data_set);
            }
        });
    };

    this._display_data_set_info_table = function (selector, data_set) {
        var template_url = 'views/data_set_info_table.mustache';
        $.get(template_url, function (template) {
            $('#' + selector).append(
                Mustache.render(template, {
                    observation: data_set.observation,
                    name: data_set.name,
                    created_at: data_set.created_at,
                    transmitter_frequency: data_set.config.transmitter_frequency,
                    sampling_time: data_set.config.sampling_time,
                    human_timestamp: data_set.config.human_timestamp,
                    nbeams: data_set.config.nbeams,
                })
            );
        });
    };

    this._plot_beam_configuration = function (selector, data_set) {
        var title = 'Multi-Beam';
        var x_label = 'RA (deg)';
        var y_label = 'DEC (deg)';

        var reference = data_set['config']['reference_pointing'];
        var beam_pointings = data_set['config']['pointings'];
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
    };

    this.plot_beam_candidates = function () {
        var data_url = self.host + "/monitoring/" + self.observation + "/" + self.data_set + "/multi_beam/beam_candidates";
        var min_freq = 409.99;
        var max_freq = 410.01;

        $.ajax({
            url: data_url,
            data: {
                min_frequency: min_freq,
                max_frequency: max_freq
            },
            success: function (beam_data) {
                var beam_candidates = beam_data['candidates'];
                var beam_firing_order = beam_data['order'];

                // Plot the beam candidates
                self._plot_beam_candidates(beam_candidates);

                // Build the beam candidates data table
                self._plot_orbit_determination_data_table(beam_candidates);

                // Plot the beam firing order
                self._plot_beam_firing_order(beam_firing_order);

                // Plot the beam illumination time-line
                self._plot_beam_illumination_timeline(beam_candidates);
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
