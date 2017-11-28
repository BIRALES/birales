function DopplerProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Beam Candidates';
    this.x_label = 'Channel (MHz)';
    this.y_label = 'Timestamp (UTC)';

    this.layout = {
        xaxis: {
            title: this.x_label
        },
        yaxis: {
            title: self.y_label,
            domain: [0.0, 0.6]
        },
        yaxis2: {
            title: 'SNR (dB)',
            domain: [0.7, 1]
        },
    };

    this.traces = [];

    this.plot = Plotly.newPlot(this.selector, this.traces, this.layout);
}


DopplerProfilePlotter.prototype = {
    constructor: DopplerProfilePlotter,

    _get_series: function (beam_candidates) {
        var series = [];

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
                name: 'beam ' + beam_candidate.beam_id + ' candidate'
            };

            if (beam_candidate['min_time'] < self._min_time) {
                self._min_time = (new Date(beam_candidate['min_time'].$date)).toISOString();
            }

            if (beam_candidate['max_time'] > self._max_time) {
                self._max_time = (new Date(beam_candidate['max_time'].$date)).toISOString();
            }

            series.push(beam_candidates_trace);
            series.push(beam_candidates_snr_trace);
        });

        return series;
    },

    update: function (beam_candidates) {
        this.traces = this._get_series(beam_candidates);

        Plotly.extendTraces(this.selector, this.traces, [0, 1]);

        log.debug('Updating the', self.name, 'plotter with', beam_candidates.length, 'new candidates');
    }
};


