Plugin.prototype.init = function () {
    var that = this;

    // Add class for default CSS stylesheet
    this.element.addClass("jquery-trackswitch");

    // Simplified main control with only play/pause and time display
    if (this.element.find(".main-control").length === 0) {
        this.element.prepend(
            '<div class="main-control">' +
                '<ul class="control">' +
                    '<li class="playpause button">Play</li>' +
                    '<li class="timing">' +
                        '<span class="time">--:--</span> / ' +
                        '<span class="length">--:--</span>' +
                    '</li>' +
                '</ul>' +
            '</div>'
        );
    }

    this.element.on('touchstart mousedown', '.playpause', $.proxy(this.event_playpause, this));
    this.element.one('loaded', $.proxy(this.loaded, this));
    this.element.one('errored', $.proxy(this.errored, this));

    this.numberOfTracks = this.element.find('ts-track').length;

    if (this.numberOfTracks > 0) {
        this.element.find('ts-track').each(function (i) {
            that.trackProperties[i] = { mute: false, solo: false, success: false, error: false };
        });

        this.updateMainControls();

        if (!audioContext) {
            this.element.trigger("errored");
            this.element.find("#overlaytext").html("Web Audio API is not supported in your browser. Please consider upgrading.");
            return false;
        }
    } else {
        this.element.trigger("errored");
    }
};

Plugin.prototype.updateMainControls = function () {
    this.element.find(".playpause").toggleClass('checked', this.playing);

    if (this.longestDuration !== 0) {
        $(this.element).find('.timing .time').html(this.secondsToHHMMSS(this.position));
        $(this.element).find('.timing .length').html(this.secondsToHHMMSS(this.longestDuration));
    }
};

Plugin.prototype.secondsToHHMMSS = function (seconds) {
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return (m < 10 ? '0' + m : m) + ':' + (s < 10 ? '0' + s : s);
};

Plugin.prototype.monitorPosition = function (context) {
    context.position = context.playing ? audioContext.currentTime - context.startTime : context.position;

    if (context.position >= context.longestDuration) {
        context.position = 0;
        context.stopAudio();

        if (context.repeat) {
            context.startAudio(context.position);
        } else {
            context.playing = false;
        }
    }

    context.updateMainControls();
};

Plugin.prototype.event_playpause = function (event) {
    event.preventDefault();

    if (!this.playing) {
        this.startAudio();
        this.playing = true;
    } else {
        this.pause();
    }

    this.updateMainControls();
    event.stopPropagation();
    return false;
};
