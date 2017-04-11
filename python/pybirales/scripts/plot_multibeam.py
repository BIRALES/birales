from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np, struct
import sys, math, os, sys
import glob, re, operator

if __name__ == "__main__":

    p = OptionParser()
    p.set_usage('process_beams.py [options] FILE_PATTERN')
    p.set_description(__doc__)

    p.add_option('-d', '--dec', dest='dec', type='float', default=0,
        help='Declination of first beam. Default value 0')
    p.add_option('', '--delta_dec', dest='delta_dec', type='float', default=1,
        help='Declination difference between beams. Default value 1')
    p.add_option('-n', '--name', dest='name', type='string', default="Cygnus",
        help='Source name. Default Cygnus')
    p.add_option('-c', '--center', dest='center', type='int', default=0,
        help='Central beam (or beam on source). Default value 0')
    p.add_option('-t', '--tsamp', dest='tsamp', type='float', default=51.2e-6,
        help='Sampling Time. Default value 51.2e-6')

    opts, args = p.parse_args(sys.argv[1:])

    # Get filenames
#    files = {}
#    for f in glob.glob(sys.argv[1]):
#        number = int(re.match(".*beam(?P<number>\d+).*", f).groupdict()['number'])
#        files[number] = f

    files  = {
    11: "/data/10_04_2017/beam_test_1491843414_beam11_timeseries_64.dat",
    12: "/data/10_04_2017/beam_test_1491843414_beam12_timeseries_64.dat",
    13: "/data/10_04_2017/beam_test_1491843414_beam13_timeseries_64.dat",
    14: "/data/10_04_2017/beam_test_1491843414_beam14_timeseries_64.dat",
    15: "/data/10_04_2017/beam_test_1491843414_beam15_timeseries_64.dat",
    16: "/data/10_04_2017/beam_test_1491843414_beam16_timeseries_64.dat",
    17: "/data/10_04_2017/beam_test_1491843414_beam17_timeseries_64.dat",
    18: "/data/10_04_2017/beam_test_1491843414_beam18_timeseries_64.dat",
    19: "/data/10_04_2017/beam_test_1491843414_beam19_timeseries_64.dat",
#    0 : "/data/27_05_2016/casa/CasA_beam0_timeseries_64.dat",
#    1 : "/data/27_05_2016/casa/CasA_beam1_timeseries_64.dat",
#    2 : "/data/27_05_2016/casa/CasA_beam2_timeseries_64.dat",
#    3 : "/data/27_05_2016/casa/CasA_beam3_timeseries_64.dat",
#    4 : "/data/27_05_2016/casa/CasA_beam4_timeseries_64.dat",
#    5 : "/data/27_05_2016/casa/CasA_beam5_timeseries_64.dat",
#    6 : "/data/27_05_2016/casa/CasA_beam6_timeseries_64.dat",
#    7 : "/data/27_05_2016/casa/CasA_beam7_timeseries_64.dat",
#    8 : "/data/27_05_2016/casa/CasA_beam8_timeseries_64.dat",
#    9 : "/data/27_05_2016/casa/CasA_beam9_timeseries_64.dat",
#    10 : "/data/27_05_2016/casa/CasA_beam10_timeseries_64.dat",
#    11 : "/data/27_05_2016/casa/CasA_beam11_timeseries_64.dat",
#    12 : "/data/27_05_2016/casa/CasA_beam12_timeseries_64.dat",
    }


    # Plot all the files
    for k, v in files.iteritems():
        data = open(v, 'rb').read()
        data = np.array(struct.unpack("f" * (len(data) / 4), data))
        #x = [ (0 + i * opts.tsamp) / 60.0 for i in range(len(data))]
        plt.plot(20*np.log10(data), label="Beam %d" % k)

    plt.legend()
    plt.title("%s transit (Beam %d on source)" % (opts.name, opts.center))
    plt.xlabel("Time since start of observation (minutes)")
    plt.ylabel("Arbitrary power")
    plt.show()
