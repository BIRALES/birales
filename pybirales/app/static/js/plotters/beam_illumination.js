function BeamIlluminationPlot(selector) {
    this.selector = selector;

    this.title = "SNR profile";
    this.x_label = 'Date';
    this.y_label = 'SNR';

    this.layout = {
        xaxis: {
            title: this.x_label
        },
        yaxis: {
            title: this.y_label,
            tickformat: ".2f"
        },
        margin: {
            l: 45,
            r: 10,
            b: 50,
            t: 20,
            pad: 4
        },
    };

    this.traces = [{
        x: [],
        y: [],
        mode: 'lines',
        name: this.title,
        line: {
            color: '#9368E9'
        }
    }];

    this.plot = Plotly.newPlot(this.selector, this.traces, this.layout);
}

BeamIlluminationPlot.prototype = {
    constructor: BeamIlluminationPlot,

    _get_series: function (beam_candidates) {
        var series = [];

        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_snr_trace = {
                x: beam_candidate['data']['time'],
                y: beam_candidate['data']['snr'],
                text: beam_candidate['data']['channel'],
                type: 'markers',
                name: 'beam ' + beam_candidate.beam_id
                // showlegend: false,
            };

            if (beam_candidate['min_time'] < self._min_time) {
                self._min_time = beam_candidate['min_time'];
            }

            if (beam_candidate['max_time'] > self._max_time) {
                self._max_time = beam_candidate['max_time'];
            }
            series.push(beam_candidates_snr_trace);
        });

        return series;
    },

    update: function (beam_candidates) {
        this.traces = this._get_series(beam_candidates);

        Plotly.newPlot(this.selector, this.traces, [0]);

        log.debug('Updating the', self.name, 'plotter with', beam_candidates.length, 'new candidates');
    }
};

