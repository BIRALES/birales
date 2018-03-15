function TrackDopplerProfilePlotter(selector) {
    this.selector = selector;
    this.title = 'Track';
    this.name = 'Doppler Profile';
    this.x_label = 'Channel (MHz)';
    this.y_label = 'Timestamp (Local Time)';
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
                type: 'time',
                time: {
                    unit: 'second',
                    displayFormats: {
                        second: 'hh:mm:ss Z'
                    }
                },
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

            $.each(tracks, function (track_id, track) {
                var dataset_data = [];
                $.each(track['data']['channel'], function (i) {
                    dataset_data.push({
                        x: track['data']['channel'][i],
                        y: moment.utc(track['data']['time'][i]['$date'])
                    })
                });

                data.datasets.push({
                    label: 'Track ' + track_id,
                    data: dataset_data
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


