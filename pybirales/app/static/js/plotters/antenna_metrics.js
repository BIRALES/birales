function AntennaMetricsPlotter(selector) {
    this.selector = selector;
    this.title = 'Antenna Metrics';
    this.name = 'Antenna Power';
    this.x_label = 'Timestamp (Local)';
    this.y_label = 'Voltage';
    this.data = {
        labels: [],
        datasets: []
    };

    this.options = {
        responsive: true,
        animation: false,
        legend: {
            position: 'bottom'
        },
        title: {
            display: false,
            text: this.title
        },
        scaleOverride: true,
        //** Required if scaleOverride is true **
        //Number - The number of steps in a hard coded scale
        scaleSteps: 10,
        //Number - The value jump in the hard coded scale
        scaleStepWidth: 10,
        //Number - The scale starting value
        scaleStartValue: new Date(),
        tooltips: {
            callbacks: {
                label: function (tooltip) {
                    let d = moment(tooltip.xLabel).toDate();
                    let date_string = d.getUTCHours() + ':' + d.getUTCMinutes() + ':' + d.getUTCSeconds();

                    return 'Antenna ' + tooltip.datasetIndex + ': ' + Math.round(tooltip.yLabel) + ' V at ' + date_string;
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

    this.plot = new Chart(document.getElementById(this.selector).getContext("2d"), {
        type: 'line',
        data: this.data,
        options: this.options
    });
}


AntennaMetricsPlotter.prototype = {
    constructor: AntennaMetricsPlotter,

    get_color: function (j) {
        return COLORS[(j + 1) % COLORS.length]
    },

    update: function (data) {
        let self = this;

        self.data.labels.push(new Date(data.timestamp));

        $.each(data.voltages, function (i, value) {
            if (self.data.datasets[i] === undefined) {
                self.data.datasets[i] = {
                    data: [],
                    borderColor: self.get_color(i),
                    backgroundColor: self.get_color(i),
                    fill: false,
                    lineTension: 0,
                    pointStrokeColor: self.get_color(i),
                    label: 'A' + i
                };
            }
            self.data.datasets[i].data.push(value);

            if (self.data.datasets[i].data.length > 30) {
                self.data.datasets[i].data.shift();
            }
        });

        self.plot.update();
    }
};

