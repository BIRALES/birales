#!/usr/bin/env python
"""
Rohde&Schwarz HMP4040 Power Supply

"""

import socket
import time
from optparse import OptionParser

__author__ = "Andrea Mattana"
__copyright__ = "Copyright 2020, Istituto di RadioAstronomia, INAF, Italy"
__credits__ = ["Andrea Mattana"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Andrea Mattana"


class RSPowerSupply(object):

    def __init__(self, ip="10.0.10.180", port=5025, timeout=2):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.connected = False

        self.supply = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.supply.settimeout(self.timeout)

        try:
            self.supply.connect((self.ip, self.port))
            print("Connected to the Power Supply (%s:%d)" % (self.ip, self.port))
            self.connected = True
        except:
            print("Unable to estabilish a connection with the RS HMP4040 Power Supply (%s:%d)" % (self.ip, self.port))

    def write(self, qst="INST OUT1\n"):
        if not qst[-1] == "\n":
            qst = qst + "\n"
        self.supply.sendall(qst)

    def read(self):
        return self.supply.recv(1024)

    def query(self, qst="*IDN?"):
        self.write(qst)
        time.sleep(0.3)
        return self.read()

    def ask_idn(self):
        self.write("*IDN?")
        time.sleep(1)
        return self.read()

    def clear_errors(self):
        self.write("SYST:ERR?")

    def reset(self):
        self.write("*RST")

    def set_voltage(self, volt):
        self.write("VOLT %3.1f" % (float(volt)))

    def set_current(self, curr):
        self.write("CURR %3.1f" % (float(curr)))

    def get_voltage(self):
        return self.query("VOLT?")

    def get_current(self):
        return self.query("CURR?")

    def apply(self, volt, curr):
        return self.write("APPLY %3.1f, %3.1f" % (float(volt), float(curr)))

    def conf_channel(self, chan, volt, curr):
        self.write("INST OUT%d" % int(chan))
        time.sleep(0.1)
        self.set_voltage(volt)
        time.sleep(0.1)
        self.set_current(curr)

    def enable_channel(self, chan):
        self.write("INST OUT%d" % int(chan))
        time.sleep(0.1)
        self.write("OUTP:SEL ON")

    def enable_channels(self, chans):
        if len(chans.split(",")):
            for c in chans.split(","):
                self.enable_channel(c)

    def disable_channel(self, chan):
        self.write("INST OUT%d" % int(chan))
        time.sleep(0.1)
        self.write("OUTP:SEL OFF")

    def is_channel_enabled(self, chan):
        self.write("INST OUT%d" % int(chan))
        time.sleep(0.1)
        if int(s.query("OUTP:SEL?")):
            return True
        else:
            return False

    def disable_channels(self, chans):
        if len(chans.split(",")):
            for c in chans.split(","):
                self.disable_channel(c)

    def enable_output(self):
        self.write("OUTP:GEN ON")

    def disable_output(self):
        self.write("OUTP:GEN OFF")

    def is_output_enabled(self):
        if int(self.query("OUTP:GEN?")):
            return True
        else:
            return False

    def get_channel_current(self, chan):
        self.write("INST OUT%d" % int(chan))
        time.sleep(0.1)
        return self.query("MEAS:CURR?")

    def get_channel_voltage(self, chan):
        self.write("INST OUT%d" % int(chan))
        time.sleep(0.1)
        return self.query("MEAS:VOLT?")

    def close(self):
        self.supply.close()
        self.connected = False


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--ip", type=str, action="store", dest="ip", default="10.0.10.180", help="IP Address")
    parser.add_option("--port", type=int, action="store", dest="port", default=5025, help="Port Address")
    parser.add_option("--action", action="store", dest="action", default="status", help="status | on | off")
    parser.add_option("--channel", action="store", dest="channel", default="ALL",
                      help="1: FRB TPM, 4: BIRALES TPM (def: ALL, you can specify comma separated channels)")
    parser.add_option("--psu_output", action="store", dest="psuoutput", default="", help="on | off")

    (opts, args) = parser.parse_args()

    s = RSPowerSupply()
    outgen = "undefined"
    if s.is_output_enabled():
        outgen = "ENABLED"
    else:
        outgen = "DISABLED"
    print("PSU Output is " + outgen)
    if opts.channel == "ALL":
        channels = range(1, 5)
    else:
        channels = [int(x) for x in opts.channel.split(",")]

    for c in channels:
        if opts.action.lower() == "status":
            voltage = float(s.get_channel_voltage(c))
            current = float(s.get_channel_current(c))
            print("Channel: %d,\tVoltage: %6.3f V,\tCurrent: %6.3f A\tPower: %3.1f W" % (
            c, voltage, current, voltage * current))
        else:
            if opts.action.lower() == "on":
                s.enable_channel(c)
            elif opts.action.lower() == "off":
                s.disable_channel(c)
            if s.is_channel_enabled(c):
                chan_stat = "ON"
            else:
                chan_stat = "OFF"
            print("Channel %d %s" % (c, chan_stat))

    if opts.psuoutput.lower() == "on":
        s.enable_output()
        print("PSU Output ENABLED")
    elif opts.psuoutput.lower() == "off":
        s.disable_output()
        print("PSU Output DISABLED")
