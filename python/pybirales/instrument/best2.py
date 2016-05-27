# Python modules
import struct
import socket
import time
import logging
import sys
import datetime
import os
import re

from pybirales.base.definitions import PipelineError


class BEST2(object):
    """ Class implementing BEST2 client to be able to move the telescope """

    def __init__(self, ip="127.0.0.1", port=7200):
        """ Class constructor """

        self._start_delay = 10
        self._movement_speed = 5  # Degrees per minute which BEST2 can move in
        self._buffer_size = 1024
        self._port = port
        self._ip = ip

        # Create socket
        self._connected = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self._socket.connect((self._ip, self._port))
            self._connected = True
        except:
            logging.warn("BEST2: Could not connect to server. Cannot move telescope")

    def get_current_declination(self):
        """ Get current BEST2 declination
        return: Current declination """

        if not self._connected:
            logging.warn("BEST2: Not connected, cannot get current declination")

        data = None
        for i in range(3):
            time.sleep(1)
            self._socket.sendall("best")  # Issue command
            time.sleep(2)      # Wait for reply
            data = self._socket.recv(self._buffer_size)  # Get reply
            if re.search("[0-9]+", data) is not None:
                break

        if re.search("[0-9]+", data) is None:
            logging.warn("BEST: Could not get current declination (got %s)" % data)
            return data
        else:
            return float(data)

    def move_to_zenith(self):
        """ Move BEST2 to zenith, which is 44.52 """
        self.move_to_declination(44.52)

    def move_to_declination(self, dec):
        """ Move BEST2 to a particular declination
        :param dec: Declination to move to
        return: Success or Failure """

        if not self._connected:
            logging.error("BEST2: Not connected, cannot get move telescope")

        # Check if declination is valid
        if not -3.0 <= dec <= 90:
            raise PipelineError("BEST2: Declination %.2f is out of range. Range is -3 to 90" % dec)

        # Check what the current declination is
        curr_declination = self.get_current_declination()
        if type(curr_declination) is not float:
            raise PipelineError("BEST: Could not get current declination (got %s)" % curr_declination)

        # Check if we need to move at all
        if abs(curr_declination - dec) < 1.0:
            logging.info("BEST2: Already in range, no need to move")
            return True

        # Check if desired declination is within 5 degrees of current declination
        if abs(curr_declination - dec) < 5.0:
            logging.info("BEST2: Current declination close to desired dec. Pointing away first")
            if curr_declination - 5.0 < -3:
                self._move_best2(curr_declination + 5.0)
            else:
                self._move_best2(curr_declination - 5.0)

        logging.info("BEST2: Pointing to desired declination (%.2f)" % dec)
        self._move_best2(curr_declination, dec)

    def _move_best2(self, current, dec):
        """ Issue the commands to move BEST2"""
        # Issue command
        self._socket.sendall("best %.2f" % dec)
        time.sleep(2)
        data = self._socket.recv(self._buffer_size)

        logging.info("BEST2: Pointing - %s" % data)

        # Wait required number of seconds
        time.sleep(self._start_delay + abs(current - dec) * self._movement_speed)

        # Check if pointing was successful
        curr_declination = self.get_current_declination()
        if type(curr_declination) is not float:
            logging.warn("BEST2: Could not validate BEST2 movement")
            return

        if abs(curr_declination - dec) < 0.5:
            return True
        else:
            raise PipelineError("BEST2: Failed to move to desired declination. Requested %.2f" % dec)
