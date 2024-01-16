$(document).ready(function () {
    // COLORS = [
    //     "#4B4E50",
    //     "#D5BBB1",
    //     "#C98CA7",
    //     "#C6D2ED",
    //     "#9CC4B2",
    //     "#E76D83",
    //     "#E7E6F7",
    //     "#827081",
    //     "#C98CA7",
    //
    //     "#0F0F0F"];

    COLORS = [

        "#ba0c3f",
        "#3cb44b",
        "#0082c8",
        "#ffe119",
        "#911eb4",
        "#f58231",
        '#4b6983',
        "#aaffc3",
        "#0082c8",
        "#808080",
        '#b39169',
        "#46f0f0",
        "#f032e6",
        "#d2f53c",
        "#cb9c9c",
        '#663822',
        "#008080",
        '#eed680',
        "#e6beff",
        '#565248',
        "#aa6e28",
        '#267726',
        "#ff5370",
        "#800000",
        '#625b81',
        '#c1665a',
        '#314e6c',
        '#d1940c',
        "#808000",
        "#000080",
        '#df421e',
        "#000000",
        '#807d74',
        '#c5d2c8'];

    Chart.defaults.global.legend.labels.boxWidth = 10;
    Chart.defaults.global.legend.labels.labelsfontSize = 10;
    Chart.defaults.global.legend.labels.labelspadding = 5;


});

var socket = io.connect('http://', {
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    reconnectionAttempts: 99999
});