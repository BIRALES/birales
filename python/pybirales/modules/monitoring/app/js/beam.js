var Beam = function (observation, data_set, beam_id) {
    var self = this;
    this.RAW_BEAM = 'raw';
    this.FILTERED_BEAM = 'filtered';
    this.HOST = "http://127.0.0.1:5000";
    this.FILTERED_BEAM_SELECTOR = 'beam-filtered-data-plot';
    this.RAW_BEAM_SELECTOR = 'beam-raw-data-plot';
    this.beam_id_selector = '#input-group-beam';
    this.freq_limits_selector = '#input-group-frequency';
    this.time_limits_selector = '#input-group-time';
    this.limit_selector_class = '.limit-control';
    this.observation_drop_down_selector = 'observations-drop_down';
    this.observation_drop_down_class_selector = '.observation-drop-down';

    this.observation_name = observation;
    this.data_set_name = data_set;
    this.beam_id = beam_id;
    this.limits = {
        min_freq: undefined,
        max_freq: undefined,
        min_time: undefined,
        max_time: undefined
    };

    this._get_beam_data = function (observation_name, data_set_name, beam_id, limits) {
        var data_url = self.HOST + "/monitoring/" + observation_name + "/" + data_set_name + "/beam/" + beam_id;
        return $.ajax({
            url: data_url,
            data: {
                min_frequency: limits.min_freq,
                max_frequency: limits.max_freq,
                min_time: limits.min_time,
                max_time: limits.max_time
            }
        });
    };

    this._get = function (object, key) {
        return $.map(object, function (a) {
            return a[key];
        });
    };

    this._plot_beam_spectrograph = function (selector, beam_candidates, beam_data, beam_id, type) {
        var title = 'Subset of Beam ' + beam_id + ' (' + type + ') data';
        var x_label = 'Channel (Mhz)';
        var y_label = 'Time (s)';

        var data = [
            {
                x: beam_data.channels,
                y: beam_data.time,
                z: beam_data.snr,
                type: 'heatmap'
            }
        ];

        $.each(beam_candidates, function (j, beam_candidate) {
            var beam_candidates_trace = {
                x: self._get(beam_candidate['detections'], 'frequency'),
                y: self._get(beam_candidate['detections'], 'time_elapsed'),
                mode: 'markers',
                name: 'beam ' + beam_candidate['beam_id'],
                marker: {
                    color: '#000000',
                    size: 6
                }
            };
            data.push(beam_candidates_trace);
        });

        var layout = {
            title: title,
            xaxis: {
                title: x_label
            },
            yaxis: {
                title: y_label
            },
            showlegend: true
        };

        Plotly.newPlot(selector, data, layout);
    };

    this._get_default_limits = function () {
        return {
            min_freq: $(self.freq_limits_selector).data('default-min-freq'),
            max_freq: $(self.freq_limits_selector).data('default-max-freq'),
            min_time: $(self.time_limits_selector).data('default-min-time'),
            max_time: $(self.time_limits_selector).data('default-max-time')
        };
    };

    this.init = function () {
        var template_url = 'views/data_sets_dropdown.mustache';
        var data_url = self.HOST + "/monitoring/observations";
        var observations = $.ajax(data_url);

        $.when(observations).done(function (observations) {
            $.get(template_url, function (template) {
                $('#' + self.observation_drop_down_selector).html(
                    Mustache.render(template, {
                        observations: observations
                    })
                );

                var default_observation = $(self.observation_drop_down_class_selector).first();
                self.observation_name = default_observation.data('data_set-observation');
                self.data_set = default_observation.data('data_set-name');

                var default_beam = $(self.beam_id_selector);
                var limits = self._get_default_limits();

                // Reload data when a new observation is selected
                $(self.observation_drop_down_selector).click(function () {
                    self.observation_name = $(this).data('data_set-observation');
                    self.data_set = $(this).data('data_set-name');
                    self.update();
                });

                // Reload data when user changes the limits
                $(self.limit_selector_class).change(function () {
                    self.update();
                });

                // Visualise the plot with default values (encoded in html)
                self.publish(default_observation.data('data_set-observation'),
                    default_observation.data('data_set-name'),
                    default_beam.data('default-beam-id'),
                    limits);
            });
        }).fail(function () {
            console.log('Failed to retrieve the available data_sets.')
        });
    };

    this.publish = function (observation_name, data_set_name, beam_id, limits) {
        var beam_data = self._get_beam_data(observation_name, data_set_name, beam_id, limits);

        $.when(beam_data).done(function (beam_data) {
            self._plot_beam_spectrograph(self.RAW_BEAM_SELECTOR, beam_data['candidates'], beam_data['raw_data'], beam_id, self.RAW_BEAM);
            self._plot_beam_spectrograph(self.FILTERED_BEAM_SELECTOR, beam_data['candidates'], beam_data['filtered_data'], beam_id, self.FILTERED_BEAM);
        });
    };

    this.update = function () {
        var limits = {};
        $(".limit-control").each(function () {
            limits[$(this).data("name")] = $(this).val();
        });

        console.log('Update: Plotting beam', beam_id, 'from', self.observation_name, self.data_set);
        this.publish(self.observation_name, self.data_set, limits.beam_id, limits)
    };
};
