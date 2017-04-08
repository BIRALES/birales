var BeamCandidatesPlotter = function () {
    var self = this;
    this.name = "Beam Candidates Plotter";
    this._debug = true;
    this.host = "http://localhost:5000";
    this._beam_candidates_url = this.host + '/monitoring/beam_candidates';

    this._beam_candidates_plot_selector = 'live-beam-candidates-plot';

    this._tx = 400.5;
    this._max_channel = 405.99;
    this._min_channel = 398.99;

    this._max_time = new Date();
    this._min_time = new Date(new Date().setHours(this._max_time.getHours() - 5));

    this._title = 'Detected Beam Candidates';
    this._x_label = 'Channel (MHz)';
    this._y_label = 'Date';

    this._get_trace = function (beam_candidates) {
        var traces = [];

        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: beam_candidate['data']['channel'],
                y: beam_candidate['data']['time'],
                z: beam_candidate['data']['snr'],
                mode: 'markers',
                name: 'beam ' + beam_candidate.beam_id + ' candidate'
            };

            if (beam_candidate['min_time'] < self._min_time) {
                self._min_time = min;
            }

            if (beam_candidate['max_time'] > self._max_time) {
                self._max_time = max;
            }

            traces.push(beam_candidates_trace);
        });

        return traces;
    };

    this._get_layout = function () {
        return {
            title: self._title,
            xaxis: {
                title: self._x_label
            },
            yaxis: {
                title: self._y_label
            },
            shapes: [
                {
                    'type': 'line',
                    'x0': self._tx,
                    'y0': self._min_time,
                    'x1': self._tx,
                    'y1': self._max_time,
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 3
                    }
                }
            ]
        };
    };

    this.publish = function () {
        var beam_candidates = self._get_beam_candidates_data(self._max_channel, self._min_channel, self._max_time.toUTCString(), self._min_time.toUTCString());

        $.when(beam_candidates).done(function (beam_candidates) {
            if (self._debug) {
                console.log('Publishing ', self.name);
            }
            var traces = self._get_trace(beam_candidates);

            var layout = self._get_layout();

            Plotly.newPlot(self._beam_candidates_plot_selector, traces, layout);

            self._update_html();
        });
    };

    this.update = function () {
        var beam_candidates = self._get_beam_candidates_data(self._max_channel, self._min_channel, self._max_time.toUTCString(), self._min_time.toUTCString());

        $.when(beam_candidates).done(function (beam_candidates) {
            if (self._debug) {
                console.log('Updating ', self.name);
            }
            var traces = self._get_trace(beam_candidates);

            var layout = self._get_layout();

            Plotly.update(self._beam_candidates_plot_selector, traces, layout);

            self._update_html();
        });
    };

    this._update_html = function () {
        $('#beam-candidates-last-updated').html('Detections since ' + self._min_time.toUTCString())
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

    // this._get = function (object, key) {
    //     return $.map(object, function (a) {
    //         return a[key];
    //     });
    // };

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
};