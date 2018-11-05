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
        pointDot: false,
        pointLabelFontSize: 50,

        legend: {
            position: 'right'
        },
        title: {
            display: false,
            text: this.title
        },
        tooltips: {
            callbacks: {
                label: function (tooltip) {
                    let d = moment(tooltip.xLabel).toDate();
                    let date_string = d.getUTCHours() + ':' + d.getUTCMinutes() + ':' + d.getUTCSeconds();

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

    plot_tracks: function (tracks) {
        let self = this;
        let data = {
            datasets: []
        };

        log.debug('Updating the', self.name, 'plotter with', tracks.length, 'new tracks');
        let beam_tracks = [];
        let n_pixels = 0;
        $.each(tracks, function (track_id, track) {
            let tx = track['tx'];
            $.each(track['data']['channel'], function (i) {
                let beam_id = track['data']['beam_id'][i];

                if (beam_tracks[beam_id] === undefined) {
                    beam_tracks[beam_id] = {
                        label: beam_id,
                        data: [],
                        pointBackgroundColor: "#ffffff",
                        pointRadius: 5,
                        borderColor: self.get_color(beam_id),
                        pointBorderColor: self.get_color(beam_id),
                        borderWidth: 2
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
            if (beam_tracks[beam_id] !== undefined) {
                data.datasets.push(beam_tracks[beam_id])
            }112
        });

        if (self.plot === undefined) {
            if (tracks.length > 0) {
                notifications.publish("Showing " + tracks.length + " tracks", 'success');
            }
            // else {
            //     notifications.publish("No detections were made", 'warning');
            // }
        }
        else {
            if (self.pixels === n_pixels) {
                self.options.animation = false;
            }
            else {
                let delta = n_pixels - self.pixels;

                if (delta > 0) {
                    notifications.publish(delta + " new detections were made", 'success');
                }
                else {
                    notifications.publish(Math.abs(delta) + " detections removed from view", 'info');
                }

            }
        }

        if (self.plot) {
            self.plot.destroy();
        }

        // Update the plot with the new data
        self.plot = new Chart(document.getElementById(self.selector).getContext("2d"), {
            type: 'scatter',
            data: data,
            options: self.options
        });

        self.pixels = n_pixels
    },

    update: function (obs_id, from_date, to_date) {
        let self = this;
        $.ajax({
            url: this.api_entry,
            method: "POST",
            data: {from_date: from_date, to_date: to_date, observation_id: obs_id}
        }).done(function (response) {
            let tracks = JSON.parse(response);
            self.plot_tracks(tracks)
        });
    }
};


