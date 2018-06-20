from wtforms import Form, StringField, DecimalField, DateTimeField, ValidationError, HiddenField
import datetime

import pytz
from wtforms import Form, StringField, DecimalField, DateTimeField, ValidationError, HiddenField
from wtforms.validators import DataRequired, AnyOf
from wtforms.widgets import TextInput


class UTCDateTimeField(DateTimeField):
    widget = TextInput()

    def __init__(self, label=None, validators=None, format='%Y-%m-%d %H:%M:%S', **kwargs):
        super(DateTimeField, self).__init__(label, validators, **kwargs)
        self.format = format

    def _value(self):
        if self.raw_data:
            return ' '.join(self.raw_data)
        else:
            return self.data and self.data.strftime(self.format) or ''

    def process_formdata(self, valuelist):
        if valuelist:
            date_str = ' '.join(valuelist)
            try:
                self.data = datetime.datetime.strptime(date_str, self.format)
                self.data = pytz.utc.localize(self.data)
            except ValueError:
                self.data = None
                raise ValueError(self.gettext('Not a valid datetime value'))


class ObservationModeForm(Form):
    obs_name = StringField('Observation name', validators=[DataRequired()])
    declination = DecimalField('Declination (&#176;)', places=4, default=45, validators=[DataRequired()])
    date_start = UTCDateTimeField('Start time', format='%Y-%m-%d %H:%M:%S',
                                  validators=[DataRequired()])
    date_end = UTCDateTimeField('End time', format='%Y-%m-%d %H:%M:%S', validators=[DataRequired()])

    def validate_date_start(self, field):
        if field.data > self.date_end.data:
            raise ValidationError('Start time should be prior to end date')

    def validate_date_end(self, field):
        if field.data < self.date_start.data:
            raise ValidationError('End date should be after start time')

    def process_data_date_end(self, field):
        a = 'a'
        print 'here'


class DetectionModeForm(ObservationModeForm):
    target_name = StringField('NORAD ID', default='UNKNOWN')
    transmitter_frequency = DecimalField('Transmitter frequency (MHz)', places=4, default=410.085,
                                         validators=[DataRequired()])


class CalibrationModeForm(ObservationModeForm):
    pass
