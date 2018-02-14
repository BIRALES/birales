var PlottingManager = function (socket) {
    // use an observer design pattern or pub/sub
    var self = this;
    var channels = {};

    this.init = function () {
        // Do initial load



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

        log.debug(plot.title, ' listening on channels ', channel);
        channels[channel].push(plot);
    };

    // Data to update
    this.updatePlots = function (channel, data) {
        log.debug('Updating plots in ' + channel + ' channel');
        $.each(channels[channel], function (channel, plot) {
            try {
                plot.update(JSON.parse(data));
            }
            catch (SyntaxError) {
                // data is already in json format, no need to parse
                plot.update(data);
            }
        })
    };
};