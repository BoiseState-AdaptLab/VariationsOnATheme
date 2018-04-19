"use strict";
var page = require('webpage').create();
var system = require('system');
var width = 1600;
var height = 900;

var url = system.args[1];
var output = system.args[2];

if (system.args.length > 4) {
    width = parseInt(system.args[3]);
    height = parseInt(system.args[4]);
}

page.viewportSize = { width: width, height: height };
page.clipRect = { top: 0, left: 0, width: width, height: height };

page.open(url, function() {
  page.render(output);
  phantom.exit();
});
