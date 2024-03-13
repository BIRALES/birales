import numpy as np
import ephem
import datetime
from pybirales.pipeline.base.definitions import CalibrationFailedException
import logging as log


class SourceDetail:

    def __init__(self, dec, time, lat, lon):
        self.source_declination = dec
        self.source_declination2 = None
        self.source_right_ascension = None
        self.source_name = None
        self.observation_start_time = time 
        log.info("Observation start time is " + str(self.observation_start_time))
        self.observing_latitude = '44:31:24.88'
        self.observing_longitude = '11:38:45.56'
        self.source_transit_time = None

        self.source_acknowledge()
        self.source_timing2()

    def source_acknowledge(self):

        log.info("Source declination {:0.3f} deg".format(self.source_declination))
        # Cassiopeia A, right ascension 23h23m24s
        if np.abs(self.source_declination - 59.) < 1.:
            log.info("Selected Cassiopeia A, right ascension 23h23m24s")
            self.source_right_ascension = '23:23:24.'
            self.source_declination2 = '58:55:27.7'

        # Taurus A, right ascension 05h34m32s
        elif np.abs(self.source_declination - 22.) < 1.:
            log.info("Selected Taurus A, right ascension 05h34m32s")
            self.source_right_ascension = '5:34:32.'
            self.source_declination2 = '22:01:32.5'

        # Virgo A, right ascension 12h30m49s
        elif np.abs(self.source_declination - 12.) < 1.:
            log.info("Selected Virgo A, right ascension 12h30m49s")
            self.source_right_ascension = '12:30:49.'
            self.source_declination2 = '12:16:54.0' 

        # Cygnus A, right ascension 19h59m28s
        elif np.abs(self.source_declination - 40.) < 1.:
            log.info("Selected Cygnus A, right ascension 19h59m28s")
            self.source_right_ascension = '19:59:28.' 
            self.source_declination2 = '40:47:23.3' 

        else:
            raise CalibrationFailedException(
                "No Calibration Source could be acknowledged at the specified declination: {:0.2f}".format(
                    self.source_declination))

    def source_timing(self):
        hours_sidereal = 23. + (56. / 60) + (4.0905 / 3600)
        sidereal_day_delta = datetime.timedelta(hours=24. - hours_sidereal)
        leap_year_multip = self.observation_start_time.year % 4 / 4.
        leap_year_delta = datetime.timedelta(hours=(24. - hours_sidereal) * leap_year_multip)
        today = self.observation_start_time.date()
        gst_0 = datetime.date(year=today.year, month=9, day=21)

        if (today - gst_0).days >= 0:
            gst_0 = datetime.date(year=today.year + 1, month=3, day=21)

        gst_today = np.abs((today - gst_0).days) * sidereal_day_delta

        longitudinal_delta = datetime.timedelta(hours=self.observing_longitude / 360. * hours_sidereal)
        lst_today = gst_today - (longitudinal_delta + sidereal_day_delta + leap_year_delta)
        # print lst_today

        ha_source = (lst_today - self.source_right_ascension).seconds
        self.source_transit_time = datetime.datetime(year=self.observation_start_time.year,
                                                     month=self.observation_start_time.month,
                                                     day=self.observation_start_time.day,
                                                     hour=0, minute=0, second=0) + datetime.timedelta(seconds=ha_source)
        # self.source_transit_time = datetime.datetime(year=2019, month=2, day=15, hour=13, minute=56, second=41)
        
    def source_timing2(self):
      
      telescope_location = ephem.Observer()
      telescope_location.lat, telescope_location.lon = self.observing_latitude, self.observing_longitude 
      telescope_location.date = self.observation_start_time
      
      source = ephem.FixedBody()
      source._ra = self.source_right_ascension
      source._dec = self.source_declination2
      source.compute(telescope_location)
      
      self.source_transit_time = source.transit_time.datetime()


