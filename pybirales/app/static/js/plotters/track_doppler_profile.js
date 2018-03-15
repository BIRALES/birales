function TrackDopplerProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Track';
    this.name = 'Doppler Profile';
    this.x_label = 'Channel (MHz)';
    this.y_label = 'Time sample';
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
        scales: {
            xAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: this.x_label
                }
            }],
            yAxes: [{
                // type: 'time',
                // time: {
                //     unit: 'second',
                //     displayFormats: {
                //         second: 'H:mm:ssZ'
                //     }
                // },
                scaleLabel: {
                    display: true,
                    labelString: this.y_label
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
                datasets: new Array(32)
            };

            log.debug('Updating the', self.name, 'plotter with', tracks.length, 'new tracks');

            $.each(tracks, function (track_id, track) {
                $.each(track['data']['channel'], function (i) {
                    var beam_id = track['data']['beam_id'][i];

                    if (data.datasets[beam_id] == undefined) {
                        data.datasets[beam_id] = {
                            label: beam_id,
                            data: [],
                            pointBackgroundColor: "#ffffff",
                            borderColor: self.get_color(beam_id),
                            pointBorderColor: self.get_color(beam_id),
                            borderWidth: 1
                        }
                    }

                    data.datasets[beam_id].data.push({
                        x: track['data']['channel'][i],
                        y: moment.utc(track['data']['time_sample'][i])
                    })
                });
            });

            // Update the plot with the new data
            self.plot = new Chart(document.getElementById(self.selector).getContext("2d"), {
                type: 'scatter',
                data: data,
                options: self.options
            });
        });
    }
};


