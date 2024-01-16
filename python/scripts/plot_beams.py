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

    filepath = ('/home/lessju/casa_observation/output/2024_01_15/Observation_2024-01-15T1501/'
                'Observation_2024-01-15T1501_beam.dat')

    nof_samples = file_size = os.path.getsize(filepath) / 4
    int_samples = 262144
    format = int_samples * 'f'
    to_plot = []
    with open(filepath, 'rb') as f:
        for i in range(int(nof_samples // int_samples)):
            data = f.read(int_samples * 4)
            data = np.array(struct.unpack(format, data))
            to_plot.append(10 * np.log10(np.sum(data) / int_samples))

    plt.plot(to_plot)
    plt.show()

    # for k, v in files.items():
    #     with
    #     data = open(v, 'rb').read()
    #     data = np.log10(np.array(struct.unpack("f" * (len(data) // 4), data)))
    #     x = [(0 + i * opts.tsamp) / 60.0 for i in range(len(data))]
    #     plt.plot(x, data, label="Beam %d" % k)
    #
    # plt.legend()
    # plt.title("%s transit (Beam %d on source)" % (opts.name, opts.center))
    # plt.xlabel("Time since start of observation (minutes)")
    # plt.ylabel("Arbitrary power")
    # plt.show()