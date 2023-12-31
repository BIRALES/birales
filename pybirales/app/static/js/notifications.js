var NotificationsManager = function () {
    this.init = function () {
        log.setLevel('debug');

        toastr.options.closeButton = true;
        toastr.options.closeMethod = 'fadeOut';
        toastr.options.closeDuration = 300;
        toastr.options.closeEasing = 'swing';
        toastr.options.showMethod = 'slideDown';
        toastr.options.hideMethod = 'slideUp';
        toastr.options.closeMethod = 'slideUp';
        toastr.options.preventDuplicates = true;
        toastr.options.positionClass = "toast-bottom-right";
    };

    this.publish = function (message, level) {
        if (level === 'error') {
            toastr.error(message, 'Error');
            log.error(message);
        }
        else if (level === 'success') {
            toastr.success(message);
            log.info(message);
        }
        else if (level === 'warning') {
            toastr.warning(message);
            log.warn(message);
        }
        else {
            toastr.info(message);
            log.info(message);
        }
    };
};


var notifications = new NotificationsManager();
notifications.init();