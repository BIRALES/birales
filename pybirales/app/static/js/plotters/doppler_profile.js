function DopplerProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Beam Candidates';
    this.x_label = 'Channel (MHz)';
    this.y_label = 'Timestamp (UTC)';

    this.layout = {
        xaxis: {
            title: this.x_label,
            domain: [0.0, 0.65]
        },
        yaxis: {
            title: this.y_label,
            domain: [0.0, 0.65]
        },
        yaxis2: {
            title: 'SNR (dB)',
            domain: [0.75, 1]
        },
        xaxis2: {
            title: 'SNR (dB)',
            domain: [0.75, 1]
        },
        margin: {
            l: 55,
            r: 10,
            b: 50,
            t: 20,
            pad: 20
        }
    };

    this.traces = [];

    this.plot = Plotly.newPlot(this.selector, this.traces, this.layout);
}


DopplerProfilePlotter.prototype = {
    constructor: DopplerProfilePlotter,

    _get_series: function (beam_candidates) {
        var series = [];

        var doppler_series = {
            x: [],
            y: [],
            mode: 'markers'
        };


        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: beam_candidate['data']['channel'],
                y: beam_candidate['data']['time'],
                text: beam_candidate['data']['snr'],
                mode: 'markers',
                name: 'beam ' + beam_candidate.beam_id
            };

            var beam_candidates_snr_trace = {
                x: beam_candidate['data']['channel'],
                y: beam_candidate['data']['snr'],
                // xaxis: 'x2',
                yaxis: 'y2',
                text: beam_candidate['data']['time'],
                mode: 'scatter',
                showlegend: false,
                name: 'beam ' + beam_candidate.beam_id
            };

             var beam_candidates_snr_trace_time = {
                x: beam_candidate['data']['snr'],
                y: beam_candidate['data']['time'],
                xaxis: 'x2',
                text: beam_candidate['data']['channel'],
                type: 'markers',
                name: 'beam ' + beam_candidate.beam_id,
                showlegend: false
            };

            if (beam_candidate['min_time'] < self._min_time) {
                self._min_time = (new Date(beam_candidate['min_time'])).toISOString();
            }

            if (beam_candidate['max_time'] > self._max_time) {
                self._max_time = (new Date(beam_candidate['max_time'])).toISOString();
            }

            series.push(beam_candidates_trace);
            series.push(beam_candidates_snr_trace);
            series.push(beam_candidates_snr_trace_time);
        });

        return series;
    },

    update: function (beam_candidates) {
        this.traces = this._get_series(beam_candidates);

        Plotly.newPlot(this.selector, this._get_series(beam_candidates), this.layout);

        log.debug('Updating the', self.name, 'plotter with', beam_candidates.length, 'new candidates');
    }
};


