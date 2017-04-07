var LivePlotter = function (observation, data_set) {
    var self = this;
    this.host = "http://localhost:5000";

    this._beam_candidates_url = this.host + '/monitoring/beam_candidates';
    this._tx = 400.5;

    this._beam_candidates_plot_selector = 'live-beam-candidates-plot';
    this._orbit_dermination_table_selector = 'live-beam-candidates-od-table';

    this._max_channel = 410.99;
    this._min_channel = 390.99;

    this._max_time = '2017-04-08 10:00:00';
    this._min_time = '2017-04-07 12:00:00';

    this._plot_beam_candidates = function (selector, beam_candidates) {
        var title = 'Detected Beam Candidates';
        var x_label = 'Channel (MHz)';
        var y_label = 'Time (s)';

        var traces = [];

        var min_time = new Date().toUTCString();
        var max_time = new Date('2017-01-01');
        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: beam_candidate['data']['channel'],
                y: beam_candidate['data']['time'],
                z: beam_candidate['data']['snr'],
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
                    'x0': self._tx,
                    'y0': min_time,
                    'x1': self._tx,
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

    // this._plot_orbit_determination_data_table = function (selector, beam_candidates, data_set) {
    //     var template_url = 'templates/includes/beam_candidate_table.mustache';
    //
    //     $.get(template_url, function (template) {
    //         $('#' + selector).empty();
    //         $.each(beam_candidates, function (candidate_number, beam_candidate) {
    //
    //             $('#' + selector).append(
    //                 Mustache.render(template, {
    //                     detections: beam_candidate.detections,
    //                     beam_id: beam_candidate.beam_id,
    //                     candidate_id: beam_candidate.name,
    //                     beam_ra: data_set.config.pointings[beam_candidate.beam_id][0],
    //                     beam_dec: data_set.config.pointings[beam_candidate.beam_id][1]
    //                 })
    //             );
    //         })
    //     });
    // };

    this.publish = function (max_channel, min_channel, max_time, min_time) {
        var beam_candidates = self._get_beam_candidates_data(max_channel, min_channel, max_time, min_time);

        $.when(beam_candidates).done(function (beam_candidates) {
            // Build the beam candidates data table
            // self._plot_orbit_determination_data_table(self._orbit_dermination_table_selector, beam_candidates_data[0]['candidates']);

            // Plot the beam candidates
            self._plot_beam_candidates(self._beam_candidates_plot_selector, beam_candidates);
        });
    };

    this._get_beam_candidates_data = function (max_channel, min_channel, max_time, min_time) {
        return $.ajax({
            url: self._beam_candidates_url,
            method: 'get',
            data: {
                max_channel: max_channel,
                min_channel: min_channel,
                max_time: max_time,
                min_time: min_time
            }
        });
    };

    this.update = function () {
        self.publish(self._max_channel, self._min_channel, self._max_time, self._min_time);
    };

    this._get = function (object, key) {
        return $.map(object, function (a) {
            return a[key];
        });
    };
};
