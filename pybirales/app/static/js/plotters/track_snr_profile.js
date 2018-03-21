function TrackSNRProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Track';
    this.name = 'SNR Profile';
    this.x_label = 'Timestamp (Local)';
    this.y_label = 'SNR (dBHz)';
    this.api_entry = '/api/live/data';
    this.color_map = colorbrewer['Set3'][12];

    this.options = {
        responsive: true,
        legend: {
            position: 'bottom'
        },
        title: {
            display: false,
            text: this.title
        },
        tooltips: {
            callbacks: {
                label: function (tooltip) {
                    var d = moment(tooltip.xLabel).toDate();
                    var date_string = d.getUTCHours() + ':' + d.getUTCMinutes() + ':' + d.getUTCSeconds();

                    return Math.round(tooltip.yLabel) + ' dBHz ,' + date_string;
                }
            }
        },
        scales: {
            xAxes: [{
                type: 'time',
                time: {
                    unit: 'second',
                    displayFormats: {
                        second: 'H:mm:ss'
                    }
                },
                scaleLabel: {
                    display: true,
                    labelString: this.x_label
                }
            }],
            yAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: this.y_label
                }
            }]
        }
    };

    this.plot = undefined;
}


TrackSNRProfilePlotter.prototype = {
    constructor: TrackSNRProfilePlotter,

    get_color: function (j) {
        return COLORS[(j + 1) % COLORS.length]
    },

    update: function (from_date, to_date) {
        var self = this;

        $.ajax({
            url: this.api_entry,
            method: "POST",
            data: {from_date: from_date, to_date: to_date}
        }).done(function (response) {
            var tracks = JSON.parse(response);
            var data = {
                datasets: []
            };

            log.debug('Updating the', self.name, 'plotter with', tracks.length, 'new tracks');
            var beam_tracks = []
            $.each(tracks, function (track_id, track) {
                $.each(track['data']['channel'], function (i) {
                    var beam_id = track['data']['beam_id'][i];

                    if (beam_tracks[beam_id] == undefined) {
                        beam_tracks[beam_id] = {
                            label: beam_id,
                            data: [],
                            fill: false,
                            borderColor: self.get_color(beam_id),
                            pointBorderColor: self.get_color(beam_id),
                            pointBackgroundColor: "#ffffff",
                            borderWidth: 1
                        }
                    }

                    beam_tracks[beam_id].data.push({
                        x: track['data']['time'][i].$date,
                        y: track['data']['snr'][i]
                    })
                });
            });

            $.each(beam_tracks, function (beam_id) {
                if (beam_tracks[beam_id] != undefined) {
                    data.datasets.push(beam_tracks[beam_id])
                }
            });

            if (self.plot != undefined) {
                self.options.animation = false;
            }

            // Update the plot with the new data
            self.plot = new Chart(document.getElementById(self.selector).getContext("2d"), {
                type: 'scatter',
                data: data,
                options: self.options
            });
        });
    }
};


