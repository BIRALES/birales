(function () {
    var d3 = Plotly.d3;

    var WIDTH_IN_PERCENT_OF_PARENT = 90,
        HEIGHT_IN_PERCENT_OF_PARENT = 90;

    var gd3 = d3.selectAll(".responsive-plot")
        .style({
            width: WIDTH_IN_PERCENT_OF_PARENT + '%',
            'margin-left': (100 - WIDTH_IN_PERCENT_OF_PARENT) / 2 + '%',

            height: HEIGHT_IN_PERCENT_OF_PARENT + '%',
            'margin-top': (100 - HEIGHT_IN_PERCENT_OF_PARENT) / 2 + '%'
        });

    var nodes_to_resize = gd3[0]; //not sure why but the goods are within a nested array

    window.onresize = function () {
        for (var i = 0; i < nodes_to_resize.length; i++) {
            Plotly.Plots.resize(nodes_to_resize[i]);
        }
    };
})();