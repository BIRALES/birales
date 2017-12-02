function DopplerProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Beam Candidates';
    this.x_label = 'Channel (MHz)';
    this.y_label = 'Timestamp (UTC)';

    this.color_map = colorbrewer['Set3'][12];

    this.layout = {
        xaxis: {
            title: this.x_label
        },
        yaxis: {
            title: this.y_label,
            domain: [0.0, 0.65]
        },
        yaxis2: {
            title: 'SNR (dB)',
            domain: [0.75, 1]
        },
        margin: {
            l: 80,
            r: 10,
            b: 50,
            t: 20,
            pad: -50
        }
    };

    this.traces = [];

    this.plot = Plotly.newPlot(this.selector, this.traces, this.layout);
}


DopplerProfilePlotter.prototype = {
    constructor: DopplerProfilePlotter,

    _get_series: function (beam_candidates) {
        var series = [];

        var c_map = this.color_map;
        $.each(beam_candidates, function (j, beam_candidate) {
            var color = c_map[(j + 1) % 12];
            var beam_candidates_trace = {
                x: beam_candidate['data']['channel'],
                y: beam_candidate['data']['time'],
                text: beam_candidate.beam_id,
                mode: 'markers',
                legendgroup: 'group_' + j,
                name: beam_candidate.beam_id,
                marker: {
                    color: color
                }
            };

            var beam_candidates_snr_trace = {
                x: beam_candidate['data']['channel'],
                y: beam_candidate['data']['snr'],
                yaxis: 'y2',
                text: beam_candidate.beam_id,
                mode: 'scatter',
                legendgroup: 'group_' + j,
                showlegend: false,
                name: beam_candidate.beam_id,
                marker: {
                    color: color
                }
            };

            if (beam_candidate['min_time'] < self._min_time) {
                self._min_time = (new Date(beam_candidate['min_time'])).toISOString();
            }

            if (beam_candidate['max_time'] > self._max_time) {
                self._max_time = (new Date(beam_candidate['max_time'])).toISOString();
            }

            series.push(beam_candidates_trace);
            series.push(beam_candidates_snr_trace);
        });

        return series;
    },

    update: function (beam_candidates) {
        this.traces = this._get_series(beam_candidates);

        Plotly.newPlot(this.selector, this._get_series(beam_candidates), this.layout);

        log.debug('Updating the', self.name, 'plotter with', beam_candidates.length, 'new candidates');
    }
};


