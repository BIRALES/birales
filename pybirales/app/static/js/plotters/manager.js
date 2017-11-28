var PlottingManager = function (socket) {
    // use an observer design pattern or pub/sub
    var self = this;
    var channels = {};

    this.init = function () {
        // Subscribe the plots to a channel
        $.each(channels, function (channel, plots) {
            socket.on(channel, function (data) {
                self.updatePlots(channel, data);
            });
        });
    };

    this.addPlot = function (channel, plot) {
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