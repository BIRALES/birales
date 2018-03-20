function TrackDopplerProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Track';
    this.name = 'Doppler Profile';
    this.y_label = 'Doppler Shift (Hz)';
    this.x_label = 'Timestamp (Local)';
    this.api_entry = '/api/live/data';
    this.color_map = colorbrewer['Set3'][12];

    this.n_points = 0;

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
                    var d = new Date(tooltip.xLabel);
                    var date_string = d.getUTCHours() + ':' + d.getUTCMinutes() + ':' + d.getUTCSeconds();

                    return Math.round(tooltip.yLabel) + ' Hz ,' + date_string;
                }
            }
        },
        scales: {
            yAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: this.y_label
                }
            }],
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
            }]
        }
    };

    this.plot = undefined;
}


TrackDopplerProfilePlotter.prototype = {
    constructor: TrackDopplerProfilePlotter,

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
            var beam_tracks = [];
            var n_pixels = 0;
            $.each(tracks, function (track_id, track) {
                var tx = track['tx'];
                $.each(track['data']['channel'], function (i) {
                    var beam_id = track['data']['beam_id'][i];

                    if (beam_tracks[beam_id] == undefined) {
                        beam_tracks[beam_id] = {
                            label: beam_id,
                            data: [],
                            pointBackgroundColor: "#ffffff",
                            borderColor: self.get_color(beam_id),
                            pointBorderColor: self.get_color(beam_id),
                            borderWidth: 1
                        }
                    }

                    beam_tracks[beam_id].data.push({
                        x: track['data']['time'][i]['$date'],
                        y: (track['data']['channel'][i] - tx) * 1e6
                    });


                });

                n_pixels += track['data']['beam_id'].length;
            });

            $.each(beam_tracks, function (beam_id) {
                if (beam_tracks[beam_id] != undefined) {
                    data.datasets.push(beam_tracks[beam_id])
                }
            });

            if (self.plot == undefined) {
                if (tracks.length > 0) {
                    notifications.publish("Showing " + tracks.length + " tracks", 'success');
                }
                else {
                    notifications.publish("No detections were made", 'warning');
                }
            }
            else {
                if (self.pixels == n_pixels) {
                    self.options.animation = false;
                }
                else {
                    notifications.publish(n_pixels - self.pixels + " new detections were made", 'success');
                }
            }

            // Update the plot with the new data
            self.plot = new Chart(document.getElementById(self.selector).getContext("2d"), {
                type: 'scatter',
                data: data,
                options: self.options
            });

            self.pixels = n_pixels
        });
    }
};


