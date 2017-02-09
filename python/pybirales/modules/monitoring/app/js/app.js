var MultiBeam = function (observation, data_set) {
    var self = this;
    this.host = "http://127.0.0.1:5000";
    this.observation_name = undefined;
    this.data_set_name = undefined;

    this._plot_beam_illumination_order = function (selector, beam_firing_order) {
        var template_url = 'views/beam_firing_order.mustache';
        $.get(template_url, function (template) {
            $('#' + selector).html(
                Mustache.render(template, {
                    order: beam_firing_order
                })
            );
        });
    };

    this._plot_beam_illumination_timeline = function (selector, beam_candidates) {

        var title = 'Beam Illumination Time line';
        var x_label = 'Time (s)';
        var y_label = 'Beam';

        var _get = function (object, key) {
            return $.map(object, function (a) {
                return a[key];
            });
        };

        var traces = [];

        $.each(beam_candidates, function (j, beam_candidate) {
            var x_data = _get(beam_candidate['detections'], 'time');
            var y_data = [];

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

    this._get = function (object, key) {
        return $.map(object, function (a) {
            return a[key];
        });
    };

    this._plot_beam_candidates = function (selector, beam_candidates, data_set) {
        var title = 'Detected Beam Candidates';
        var x_label = 'Channel (MHz)';
        var y_label = 'Time (s)';

        var traces = [];

        var min_time = new Date().toUTCString();
        var max_time = data_set.config.human_timestamp;
        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: self._get(beam_candidate['detections'], 'frequency'),
                y: self._get(beam_candidate['detections'], 'time'),
                z: self._get(beam_candidate['detections'], 'snr'),
                mode: 'markers',
                name: 'beam ' + beam_candidate.beam_id + ' candidate ' + beam_candidate.name
            };

            var min = beam_candidates_trace.y.reduce(function (a, b) {
                return a < b ? a : b;
            });
            var max = beam_candidates_trace.y.reduce(function (a, b) {
                return a > b ? a : b;
            });

            if (min < min_time) {
                min_time = min;
            }

            if (max > max_time) {
                max_time = max;
            }

            traces.push(beam_candidates_trace);
        });

        var layout = {
            title: title,
            xaxis: {
                title: x_label
            },
            yaxis: {
                title: y_label
            },
            shapes: [
                {
                    'type': 'line',
                    'x0': data_set.config.transmitter_frequency,
                    'y0': min_time,
                    'x1': data_set.config.transmitter_frequency,
                    'y1': max_time,
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 3
                    }
                }
            ]
        };

        Plotly.newPlot(selector, traces, layout);
    };

    this._plot_orbit_determination_data_table = function (selector, beam_candidates, data_set) {
        var template_url = 'views/beam_candidate_table.mustache';

        $.get(template_url, function (template) {
            $('#' + selector).empty();
            $.each(beam_candidates, function (candidate_number, beam_candidate) {

                $('#' + selector).append(
                    Mustache.render(template, {
                        detections: beam_candidate.detections,
                        beam_id: beam_candidate.beam_id,
                        candidate_id: beam_candidate.name,
                        beam_ra: data_set.config.pointings[beam_candidate.beam_id][0],
                        beam_dec: data_set.config.pointings[beam_candidate.beam_id][1]
                    })
                );
            })
        });
    };

    this._display_data_set_info_table = function (selector, data_set) {
        var template_url = 'views/data_set_info_table.mustache';
        $.get(template_url, function (template) {
            $('#' + selector).html(
                Mustache.render(template, {
                    observation: data_set.observation,
                    name: data_set.name,
                    created_at: data_set.created_at,
                    transmitter_frequency: data_set.config.transmitter_frequency,
                    sampling_time: data_set.config.sampling_time,
                    human_timestamp: data_set.config.human_timestamp,
                    nbeams: data_set.config.nbeams
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

            reference[0] = 0;
            reference[1] = 0;

            // Specify the Beam cross-sectional shapes
            var shape = {
                type: 'circle',
                xref: 'x',
                yref: 'y',
                x0: reference[0] + ra - bw_ra / 2.,
                x1: reference[0] + ra + bw_ra / 2.,
                y0: reference[1] + dec - bw_dec / 2.,
                y1: reference[1] + dec + bw_dec / 2.
            };

            // Write Beam id at the centroid of the beam cross-section
            data[0].x.push(reference[0] + ra);
            data[0].y.push(reference[1] + dec);
            data[0].text.push(beam_id);

            shapes.push(shape);
        });

        var layout = {
            title: title,
            xaxis: {
                title: x_label,
                zeroline: false
            },
            yaxis: {
                title: y_label,
                zeroline: false
            },
            width: 500,
            height: 500,
            shapes: shapes
        };

        Plotly.newPlot(selector, data, layout);
    };

    this._plot_beam_configuration_with_illumination = function (selector, beam_candidates, data_set) {
        var title = 'Multi-Beam Illumination Time line';
        var x_label = 'RA (deg)';
        var y_label = 'DEC (deg)';
        var z_label = 'Time (s)';
        var beam_pointings = data_set['config']['pointings'];
        var traces = [];
        var _get = function (object, key) {
            return $.map(object, function (a) {
                return a[key];
            });
        };

        var min_time = 100;
        $.each(beam_candidates, function (j, beam_candidate) {
            var len = beam_candidate['detections'].length;
            var beam_id = beam_candidate.beam_id;
            var ra = beam_pointings[beam_id][0];
            var dec = beam_pointings[beam_id][1];
            var beam_candidates_trace = {
                x: new Array(len).fill(ra),
                y: new Array(len).fill(dec),
                z: _get(beam_candidate['detections'], 'time_elapsed'),
                type: 'scatter3d',
                hoverinfo:'none',
                name: 'Candidate ' + beam_candidate.name
            };

            if (beam_candidate.illumination_time < min_time){
                min_time = beam_candidate.illumination_time;
            }
            traces.push(beam_candidates_trace);
        });

        $.each(beam_pointings, function (i, beam_pointing) {
            var ra = beam_pointing[0];
            var dec = beam_pointing[1];
            var pointing = {
                z: [min_time],
                x: [ra],
                y: [dec],
                type: 'scatter3d',
                text: i,
                mode: 'markers+text',
                showlegend: false,
                size: 30,
                hoverinfo:'none',
                line: {
                    width: 20,
                    color: '#666666'
                },
                textfont: {
                    family: 'sans serif',
                    size: 10,
                    color: '#000000',
                }
            };
            traces.push(pointing);
        });


        var layout = {
            shapes: [],
            scene: {
                aspectratio: {
                    x: 1,
                    y: 1,
                    z: 1
                },
                xaxis: {
                    title: x_label,
                    range: [-4, 4]
                },
                yaxis: {
                    title: y_label
                },
                zaxis: {
                    title: z_label
                }
            },
            title: title,
            xaxis: {
                autorange: true,
                title: x_label
            },
            yaxis: {
                autorange: true,
                title: y_label
            },
            zaxis: {
                autorange: true,
                title: z_label
            },
            font: {outlineColor: '#000000'},
            width: 700,
            height: 700
        };

        Plotly.newPlot(selector, traces, layout);
    };

    this.publish = function (observation_name, data_set_name) {
        this.observation_name = observation_name;
        this.data_set_name = data_set_name;

        var beam_candidates = self._get_beam_candidates_data(observation_name, data_set_name);
        var data_set = self._get_data_set_data(observation_name, data_set_name);

        // Update heading
        self._update_heading(observation_name, data_set_name);

        $.when(beam_candidates).done(function (beam_candidates_data) {
            // Plot the beam firing order
            self._plot_beam_illumination_order('beam-firing-order', beam_candidates_data['order']);

            // Plot the beam illumination time-line
            self._plot_beam_illumination_timeline('beam-illumination-time-line-plot', beam_candidates_data['candidates']);
        });

        $.when(data_set).done(function (data_set_data) {
            // Display the Data set information table
            self._display_data_set_info_table('data-set-info-table', data_set_data);

            // Display the beam plot configuration
            self._plot_beam_configuration('multi-beam-configuration-plot', data_set_data);
        });

        $.when(beam_candidates, data_set).done(function (beam_candidates_data, data_set_data) {
            // Build the beam candidates data table
            self._plot_orbit_determination_data_table('orbit-determination-data-table', beam_candidates_data[0]['candidates'], data_set_data[0]);

            // Plot the beam candidates
            self._plot_beam_candidates('beam-candidates-plot', beam_candidates_data[0]['candidates'], data_set_data[0]);

            // Display the beam plot configuration with illumination
            self._plot_beam_configuration_with_illumination('multi-beam-configuration-illumination-timeline-plot', beam_candidates_data[0]['candidates'], data_set_data[0]);
        });

        var beam_id = 6;
        var beam_raw_data = self._get_beam_raw_data(observation_name, data_set_name, beam_id);
        $.when(beam_raw_data).done(function (beam_raw_data) {
            // Plot the spectrograph of the raw beam data
            self._plot_raw_beam_spectrograph('beam-raw-data-plot', beam_raw_data['candidates'], beam_raw_data['raw_data'], beam_id);
        });
    };

    this.update = function (observation_name, data_set_name) {
        if (this.observation_name !== observation_name && this.data_set_name !== data_set_name) {
            this.publish(observation_name, data_set_name)
        }
    };

    this._get_beam_candidates_data = function (observation_name, data_set_name) {
        var data_url = self.host + "/monitoring/" + observation_name + "/" + data_set_name + "/multi_beam/beam_candidates";
        var min_freq = 409.99;
        var max_freq = 410.01;

        return $.ajax({
            url: data_url,
            data: {
                min_frequency: min_freq,
                max_frequency: max_freq
            }
        });
    };

    this._get_beam_raw_data = function (observation_name, data_set_name, beam_id) {
        var data_url = self.host + "/monitoring/" + observation_name + "/" + data_set_name + "/beam/" + beam_id;
        var min_freq = 409.990;
        var max_freq = 409.9926;

        var min_time = 20.;
        var max_time = 55.;

        return $.ajax({
            url: data_url,
            data: {
                min_frequency: min_freq,
                max_frequency: max_freq,
                min_time: min_time,
                max_time: max_time
            }
        });
    };

    this._get_data_set_data = function (observation_name, data_set_name) {
        var data_url = self.host + "/monitoring/" + observation_name + "/" + data_set_name + "/about";
        return $.ajax({
            url: data_url
        });
    };

    this._update_heading = function (observation_name, data_set_name) {
        var template_url = 'views/main_heading.mustache';
        var selector = 'observation-heading';
        $.get(template_url, function (template) {
            $('#' + selector).html(
                Mustache.render(template, {
                    observation: observation_name,
                    data_set: data_set_name
                })
            );
        });
    };

    this._plot_raw_beam_spectrograph = function (selector, beam_candidates, raw_beam_data, beam_id) {
        var title = 'Beam ' + beam_id + ' raw data';
        var x_label = 'Channel (Mhz)';
        var y_label = 'Time (s)';
        var z_label = 'SNR';

        var data = [
            {
                x: raw_beam_data.channels,
                y: raw_beam_data.time,
                z: raw_beam_data.snr,
                type: 'heatmap'
            }
        ];

        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: self._get(beam_candidate['detections'], 'frequency'),
                y: self._get(beam_candidate['detections'], 'time_elapsed'),
                mode: 'markers',
                name: 'beam ' + beam_candidate['beam_id'],
                marker: {
                    color: '#000000',
                    size: 6
                }
            };
            data.push(beam_candidates_trace);
        });

        var layout = {
            title: title,
            xaxis: {
                title: x_label
            },
            yaxis: {
                title: y_label
            },
            zaxis: {
                title: z_label
            },
            showlegend: true
        };

        Plotly.newPlot(selector, data, layout);
    };

    this.init = function () {
        var template_url = 'views/data_sets_dropdown.mustache';
        var data_url = self.host + "/monitoring/observations";
        var selector = 'observations-drop_down';
        var observation_drop_down_selector = '.observation-drop-down';
        var observations = $.ajax(data_url);

        $.when(observations).done(function (observations) {
            $.get(template_url, function (template) {
                $('#' + selector).html(
                    Mustache.render(template, {
                        observations: observations
                    })
                );

                $(observation_drop_down_selector).click(function () {
                    self.update($(this).data('data_set-observation'), $(this).data('data_set-name'));
                });

                var default_observation = $(observation_drop_down_selector).first();
                self.publish(default_observation.data('data_set-observation'), default_observation.data('data_set-name'));
            });
        }).fail(function () {
            console.log('Failed to retrieve the available data_sets.')
        });
    };
};