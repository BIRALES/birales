from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np, struct
import sys, math, os, sys
import glob, re, operator
import matplotlib

def extract_values(values):
    """ Extract values from string representation of list
    :param values: String representation of values
    :return: List of values
    """

    # Return list
    converted = []

    # Loop over all comma separated values
    for item in values.split(","):
        # Check if item contains a semi-colon
        if item.find(":") > 0:
            index = item.find(":")
            lower = item[:index]
            upper = item[index+1:]
            converted.extend(range(int(lower), int(upper) + 1))
        else:
            converted.append(int(item))

    return converted

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
    p.add_option('-c', '--center', dest='center', type='int', default=15,
        help='Central beam (or beam on source). Default value 0')
    p.add_option('-t', '--tsamp', dest='tsamp', type='float', default=0.0131072,
        help='Sampling Time. Default value 0.0131072')
    p.add_option('-b', '--nbeams', dest='nbeams', type='int', default=32,
        help='Number of beams. Default 32')
    p.add_option('-i', '--integrations', dest='integrations', type='int', default=64,
        help='Number of integrations. Default 64')
    p.add_option('-p', '--to-plot', dest='to_plot', default='15',
        help='Beams to plot. Default 15')

    opts, args = p.parse_args(sys.argv[1:])

    if len(sys.argv) < 2:
        print("File pattern required")
        exit(0)

    # Extract beams to plot
    opts.to_plot = extract_values(opts.to_plot)

    # Generate filenames
    files = {}
    for i in range(opts.nbeams):
        files[i] = "{}_beam{}_timeseries_{}.dat".format(args[0], i, opts.integrations)

    markers = matplotlib.markers.MarkerStyle.markers.keys()

    # Plot required beams
    for i, beam in enumerate(opts.to_plot):
        data = open(files[beam], 'rb').read()
        data = np.array(struct.unpack("f" * (len(data) / 4), data))
        plt.plot(np.arange(0, len(data)) * opts.tsamp / 60, 10 * np.log10(data), label="Beam %d" % beam)#, marker=markers[i])

    plt.legend()
    plt.title("%s transit (Beam %d on source)" % (opts.name, opts.center))
    plt.xlabel("Time since start of observation (minutes)")
    plt.ylabel("Arbitrary power")
    plt.show()
