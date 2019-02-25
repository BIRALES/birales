import numpy as np
import h5py
import datetime

import source_detail


class TransitSelect:

    def __init__(self, cal_input, vis_file):
        self.vis_file = vis_file
        self.cal_input = cal_input
        self.visibilities = None
        self.vis_in = None
        self.vis_peak = None
        self.source_transit_time = None

        self.h5_reader()
        self.transit_peak_find()
        print 'Transit peak search located transit at time sample ' + str(self.vis_peak) + ' at time: ' + str(self.source_transit_time)
        self.transit_compute()
        print 'Transit time computation located transit at time sample ' + str(self.vis_peak) + ' at time: ' + str(self.source_transit_time)
        self.vis_in = self.visibilities[self.vis_peak, :]

    def h5_reader(self):

        f = h5py.File(self.vis_file, "r")
        self.visibilities = np.array(f["Vis"])[:, 0, :, 0]

    def transit_compute(self):

        dec = self.cal_input['pointing_dec']
        utc_time = datetime.datetime.strptime(self.cal_input['obs_time'], '%Y-%m-%d %H:%M:%S.%f')
        time = utc_time + datetime.timedelta(hours=1.)
        lat = self.cal_input['latitude']
        lon = self.cal_input['longitude']
        integration_time = self.cal_input['integration_time']

        source_calculate = source_detail.SourceDetail(dec, time, lat, lon)
        self.source_transit_time = source_calculate.source_transit_time
        time_diff = (self.source_transit_time - time).seconds
        self.vis_peak = np.int(time_diff/integration_time)

    def transit_peak_find(self):
        """
        Only to be used when the peak visibilities in a transit observation needs to be found, prior to carrying out
        SelfCal on the peak visibilities only. Returns a vector of peak visibilities that can be fed to selfcal_run.
        This is not envisaged to be required for AAVS, except for testing purposes.

        """

        # Define data
        obs_in = np.array(self.visibilities)
        data = obs_in[:, :]
        # Define arrays to be filled
        all_peak = np.zeros(data.shape[1])

        # Start loop across baselines
        for i in range(data.shape[1]):
            # Initialize list for taking in final data points with removed outliers
            final_list = list()
            # Find power
            data1 = np.sqrt((np.abs(data[:, i].real) ** 2) + (np.abs(data[:, i].imag) ** 2))
            # Remove zero power padding regions
            data1 = np.trim_zeros(data1)
            # Find mean power
            mn = np.mean(data1, axis=0)
            # Find power std dev
            sd = np.std(data1, axis=0)
            # Start loop per baseline across time
            for j in range(data1.shape[0]):
                # Select data which falls within std dev away from mean
                if (data1[j] > mn - 2 * sd) and (data1[j] < mn + 2 * sd):
                    final_list.append(data1[j])
                # Data not within std dev range is an outlier, change with mean
                if data1[j] > mn + 2 * sd:
                    final_list.append(mn)
                if data1[j] < mn - 2 * sd:
                    final_list.append(mn)
            # Make list of final power values an array
            final_list = np.array(final_list)
            # Find time index of max power for this baseline
            all_peak[i] = np.int(np.where(final_list == np.max(final_list))[0][0])
        # Take mean peak
        self.vis_peak = np.int(np.mean(all_peak))
        integration_time = self.cal_input['integration_time']
        utc_time = datetime.datetime.strptime(self.cal_input['obs_time'], '%Y-%m-%d %H:%M:%S.%f')
        self.source_transit_time = utc_time + datetime.timedelta(seconds=3600 + (integration_time*self.vis_peak))
