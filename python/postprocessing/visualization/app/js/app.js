new Vue({
    el: '#app',

    data: {
        beams_configuration: false,
        beams_candidates: false,
        error: false
    },

    ready: function () {
        this.$http({url: '/json/data.json', method: 'GET'}).then(function (response) {
            this.$set('beams_candidates', response.beams)
        }, function (response) {
            this.$set('error', true)
        });
    }
});