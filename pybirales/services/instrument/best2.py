import socket
import threading
import time
import logging
import re

from pybirales.pipeline.base.definitions import PipelineError
from pybirales.utilities.singleton import Singleton
from pybirales.services.instrument.best2_server import SdebTCPServer


@Singleton
class BEST2(object):
    """ Class implementing BEST2 client to be able to move the telescope """

    def __init__(self, ip="127.0.0.1", port=7200):
        """ Class constructor """

        self._zenith = 44.52
        self._start_delay = 15
        self._movement_speed = 4  # Degrees per minute which BEST2 can move in
        self._buffer_size = 1024
        self._port = port
        self._ip = ip

        # Server thread
        self._best2_server = None
        self._stop_server = False

        # Create socket
        self._connected = False
        self._socket = None

        # Connect to server
        self._connect()

    def _connect(self):
        """ Connect to server """
        try:
            # Check if server is already running
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._ip, self._port))
            self._connected = True
            logging.info("Connected to existing backend server")
        except socket.error:
            # If not launch server in a separate thread
            logging.info("Launching new backend server")
            self._thread = threading.Thread(target=self.start_best2_server)
            self._thread.start()

            time.sleep(1)

            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._ip, self._port))
            self._connected = True

        # Keep track of current pointing
        self.get_current_declination()

    def start_best2_server(self):
        """ Start BEST-2 server"""
        self._best2_server = SdebTCPServer(("", self._port))
        while not self._stop_server:
            self._best2_server.handle_request()
        self._stop_server = False
        logging.info("BEST-II server stopped")

    def stop_best2_server(self):
        """ Stop BEST-2 server """
        if self._connected:
            logging.info("Stopping BEST-II server")
            self._socket.close()
            if self._best2_server is not None:
                self._stop_server = True
                while self._stop_server:
                    logging.info("Waiting for BEST-II server to stop")
                    time.sleep(1)

                self._best2_server.server_close()
                del self._best2_server
                self._best2_server = None
            self._connected = False

    def get_current_declination(self):
        """ Get current BEST2 declination
        return: Current declination """

        if not self._connected:
            logging.warn("BEST2 not connected, connecting")
            self._connect()

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
            raise PipelineError("BEST: Could not get current declination (got %s)" % data)
        else:
            self.current_pointing = float(data)
            return float(data)

    def move_to_zenith(self):
        """ Move BEST2 to zenith, which is 44.52 """
        if self.move_to_declination(self._zenith):
            logging.info("BEST2: Pointed to declination (%.2f)" % self.current_pointing)
            return True
        else:
            return False

    def move_to_declination(self, dec):
        """ Move BEST2 to a particular declination
        :param dec: Declination to move to
        return: Success or Failure """

        if not self._connected:
            logging.warn("BEST2 not connected, connecting")
            self._connect()

        # Check if declination is valid
        if not -3.0 <= dec <= 90:
            raise PipelineError("BEST2: Declination %.2f is out of range. Range is -3 to 90" % dec)

        # Check if we need to move at all
        if abs(self.current_pointing - dec) < 1.0:
            logging.info("BEST2: Already in range, no need to move")
            return True

        # Check if desired declination is within 5 degrees of current declination
        if abs(self.current_pointing - dec) < 5.0:
            logging.info("BEST2: Current declination close to desired dec. Pointing away first")
            if self.current_pointing - 5.0 < 0:
                self._move_best2(self.current_pointing + 5.0)
            else:
                self._move_best2(self.current_pointing - 5.0)

        logging.info("BEST2: Pointing to desired declination (%.2f)" % dec)
        if self._move_best2(dec):
            logging.info("BEST2: Pointed to declination (%.2f)" % self.current_pointing)
            return True
        else:
            return False

    def pointing_time(self, dec1, dec2):
        """ Calculate the time required to move between two positions """
        return self._start_delay + abs(dec1 - dec2) * self._movement_speed

    def _move_best2(self, dec):
        """ Issue the commands to move BEST2"""
        # Issue command
        self._socket.sendall("best %.2f" % dec)
        time.sleep(2)
        data = self._socket.recv(self._buffer_size)

        logging.info("BEST2: Pointing - %s" % data)

        # Wait for start delay
        time.sleep(self._start_delay)

        # Wait until pointing is very close to desired one
        while True:
            self._socket.sendall("progress")
            try:
                data = self._socket.recv(self._buffer_size)
                value = float(data.split("   ")[2])
                self.current_pointing = value
                logging.info("Current pointing: {}".format(self.current_pointing))

                if abs(value - dec) < 1.5:
                    # We are ready
                    time.sleep(2)
                    break
            except:
                pass

        # Check if pointing was successful
        curr_declination = self.get_current_declination()
        if type(curr_declination) is not float:
            logging.warning("BEST2: Could not validate BEST2 movement")
            return

        if abs(curr_declination - dec) < 0.5:
            return True
        else:
            raise PipelineError("BEST2: Failed to move to desired declination. Requested %.2f" % dec)


if __name__ == "__main__":
    # Test out BEST2 pointing

    # Set logging
    from sys import stdout
    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    str_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(stdout)
    ch.setFormatter(str_format)
    log.addHandler(ch)

    best2 = BEST2.Instance()

    best2.stop_best2_server()

    best2 = BEST2.Instance()

    best2.move_to_declination(30)

    best2.stop_best2_server()

