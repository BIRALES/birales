var BeamCandidatesPlotter = function () {
    var self = this;
    this.name = "Beam Candidates";
    this.host = "";
    this._beam_candidates_url = this.host + '/monitoring/beam_candidates';

    this._beam_candidates_plot_selector = 'live-beam-candidates-plot';
    this._beam_candidates_plot_selector_id = '#' + this._beam_candidates_plot_selector;

    this._tx = $(this._beam_candidates_plot_selector_id).data('chart-tx') || 410.0703125;
    this._max_channel = $(this._beam_candidates_plot_selector_id).data('chart-max_channel') || 410.0703125;
    this._min_channel = $(this._beam_candidates_plot_selector_id).data('chart-min_channel') || 398.9921875;

    this._max_time = $(this._beam_candidates_plot_selector_id).data('chart-max_time');
    this._min_time = $(this._beam_candidates_plot_selector_id).data('chart-min_time');

    this._title = '';
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
                self._min_time = beam_candidate['min_time'];
            }

            if (beam_candidate['max_time'] > self._max_time) {
                self._max_time = beam_candidate['max_time'];
            }

            traces.push(beam_candidates_trace);
        });

        return traces;
    };

    this._get_layout = function () {
        log.debug('Time Range: ', self._min_time, self._max_time);
        return {
            title: self._title,
            xaxis: {
                title: self._x_label
            },
            yaxis: {
                title: self._y_label
            },
            margin: {
                l: 120,
                r: 50,
                b: 50,
                t: 20,
                pad: 10
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
        var beam_candidates = self._get_beam_candidates_data(self._max_channel, self._min_channel, self._max_time, self._min_time);

        $.when(beam_candidates).done(function (beam_candidates) {
            if(beam_candidates){
                log.debug('Publishing the', self.name, 'plotter with', beam_candidates.length,'candidates');
                var traces = self._get_trace(beam_candidates);

                var layout = self._get_layout();

                Plotly.newPlot(self._beam_candidates_plot_selector, traces, layout);

                self._update_html();
            }
            else{
                log.debug('No candidates were detected');
            }
        });
    };

    this.update = function () {
        var from = self._max_time;
        var beam_candidates = self._get_beam_candidates_data(self._max_channel, self._min_channel, undefined, from);

        $.when(beam_candidates).done(function (beam_candidates) {
            if(beam_candidates){
                var traces = self._get_trace(beam_candidates);

                var layout = self._get_layout();

                Plotly.update(self._beam_candidates_plot_selector, traces, layout);

                self._update_html();

                log.debug('Updating the', self.name, 'plotter with', beam_candidates.length, 'new candidates');
            }
            else {
                log.debug('Nothing to update. No new candidates were detected');
            }
        });
    };

    this._update_html = function () {
        $('#beam-candidates-last-updated').html('Detections since ' + self._min_time)
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
