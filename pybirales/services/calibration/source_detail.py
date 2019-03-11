import numpy as np
import datetime


class SourceDetail:

    def __init__(self, dec, time, lat, lon):
        self.source_declination = dec
        self.source_right_ascension = None
        self.source_name = None
        self.observation_start_time = time
        self.observing_latitude = lat
        self.observing_longitude = lon
        self.source_transit_time = None

        self.source_acknowledge()
        self.source_timing()

    def source_acknowledge(self):
        # Cassiopeia A, right ascension 23h23m24s
        if np.abs(self.source_declination - 59.) < 1.:
            self.source_right_ascension = datetime.timedelta(hours=(23. + (23./60) + (24./3600)))

        # Taurus A, right ascension 05h34m32s
        if np.abs(self.source_declination - 22.) < 1.:
            self.source_right_ascension = datetime.timedelta(hours=(5. + (34./60) + (32./3600)))

        # Virgo A, right ascension 12h30m49s
        if np.abs(self.source_declination - 12.) < 1.:
            self.source_right_ascension = datetime.timedelta(hours=(12. + (30./60) + (49./3600)))

        # Cygnus A, right ascension 19h59m28s
        if np.abs(self.source_declination - 40.) < 1.:
            self.source_right_ascension = datetime.timedelta(hours=(19. + (59./60) + (28./3600)))

    def source_timing(self):
        hours_sidereal = 23. + (56./60) + (4.0905/3600)
        sidereal_day_delta = datetime.timedelta(hours=24. - hours_sidereal)
        leap_year_multip = self.observation_start_time.year % 4 / 4.
        leap_year_delta = datetime.timedelta(hours=(24. - hours_sidereal)*leap_year_multip)
        today = self.observation_start_time.date()
        gst_0 = datetime.date(year=today.year, month=9, day=20)
        if (today - gst_0).days >= 0:
            gst_0 = datetime.date(year=today.year+1, month=9, day=20)
        gst_today = np.abs((today - gst_0).days) * sidereal_day_delta

        longitudinal_delta = datetime.timedelta(hours=self.observing_longitude/360. * hours_sidereal)
        lst_today = gst_today - (longitudinal_delta + sidereal_day_delta + leap_year_delta)

        ha_source = (lst_today - self.source_right_ascension).seconds
        self.source_transit_time = datetime.datetime(year=self.observation_start_time.year,
                                                     month=self.observation_start_time.month,
                                                     day=self.observation_start_time.day,
                                                     hour=0, minute=0, second=0) + datetime.timedelta(seconds=ha_source)
        # self.source_transit_time = datetime.datetime(year=2019, month=2, day=15, hour=13, minute=56, second=41)
