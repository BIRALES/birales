from datetime import datetime, timedelta

from wtforms import Form, StringField, DecimalField, DateTimeField, DateField, ValidationError
from wtforms.validators import DataRequired


class ObservationModeForm(Form):
    obs_name = StringField('Observation name', validators=[DataRequired()])
    declination = DecimalField('Declination (&#176;)', places=4, default=45, validators=[DataRequired()])
    date_start = DateTimeField('Start time (UTC)', format='%Y-%m-%d %H:%M:%S',
                               validators=[DataRequired()])
    date_end = DateTimeField('End time (UTC)', format='%Y-%m-%d %H:%M:%S', validators=[DataRequired()])

    def validate_date_start(self, field):
        if field.data > self.date_end.data:
            raise ValidationError('Start time should be prior to end date')

    def validate_date_end(self, field):
        if field.data < self.date_start.data:
            raise ValidationError('End date should be after start time')


class DetectionModeForm(ObservationModeForm):
    target_name = StringField('NORAD ID', default='UNKNOWN')
    transmitter_frequency = DecimalField('Transmitter frequency (MHz)', places=4, default=410.085,
                                         validators=[DataRequired()])


class CalibrationModeForm(ObservationModeForm):
    pass
