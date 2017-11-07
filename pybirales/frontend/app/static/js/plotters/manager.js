var PlottingManager = function (socket) {
    // use an observer design pattern or pub/sub
    var self = this;
    var channels = {};
    var initialised = false;

    this.init = function () {
        // Add plots here
         new BeamCandidatesPlotter(),
                new BeamCandidatesSNRProfilePlotter()
        self.addPlot(new RawDataTablePlot('measurement-set-table'), 'measurements');
        self.addPlot(new RadiationPatternPlot('radiation-pattern-plot'), 'measurements');
        self.addPlot(new DroneAltitudePlot('drone-altitude-plot'), 'measurements');
        self.addPlot(new BandPassPlot('bandpass-plot'), 'measurements');
        self.addPlot(new PowerSpectrumPlot('power-spectrum-plot'), 'trace');

        // Update plots
        $.each(channels, function (channel) {
            socket.on(channel, function (data) {
                self.updatePlots(channel, data);
            });
        });

        initialised = true;
    };

    this.addPlot = function (plot, channel) {
        if (channels[channel] == undefined) {
            channels[channel] = [];
        }

        channels[channel].push(plot);
    };

    // Data to update
    this.updatePlots = function (channel, data) {
        log.debug('Updating plots in ' + channel + ' channel');
        $.each(channels[channel], function (channel, plot) {
            plot.update(data);
        })
    };
};