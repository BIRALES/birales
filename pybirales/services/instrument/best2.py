import logging
import re
import socket
import threading
import time

from pybirales.pipeline.base.definitions import BEST2PointingException
from best2_server import SdebTCPServer, Pointing
from pybirales.utilities.singleton import Singleton

# Delay in seconds
START_DELAY = 15

# Degrees per minute which BEST2 can move in
MOVEMENT_SPEED = 4


def pointing_time(dec1, dec2):
    """
    Calculate the time required to move between two positions

    :param dec1: original declination
    :param dec2: target declination
    :return:
    """
    return START_DELAY + abs(dec1 - dec2) * 60 / MOVEMENT_SPEED


@Singleton
class BEST2(object):
    """
    Class implementing BEST2 client to be able to move the telescope
    """

    # The acceptable error between desired and current pointing
    DEC_ERROR_THOLD = 0.75

    def __init__(self, ip="192.168.30.134", port=5002):
        """ Class constructor """
        ip = "127.0.0.1"
        port = 7200
        self._zenith = 44.52
        self._buffer_size = 1024
        self._port = port
        self._ip = ip

        # Server thread
        self._best2_server = None
        self._stop_server = False

        # Create socket
        self._connected = False
        self._socket = None

    def connect(self, tries=0):
        """
        Connect to the BEST-II server

        :return:
        """
        tries += 1
        if tries > 3:
            raise BEST2PointingException("Could not connect to antenna server. Number of tries exceeded")
        
        
        try:
            # Check if server is already running
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(20)
            self._socket.connect((self._ip, self._port))
        except socket.timeout:
            # If not launch server in a separate thread
            logging.warning("Timeout. BEST antenna server is not available. Launching new BEST antenna server")

            self._launch_new_server()

            # Try to reconnect
            self.connect(tries)
        except socket.error:
            # If not launch server in a separate thread
            self._launch_new_server()

            # Try to reconnect
            self.connect(tries)
        else:
            self._connected = True
            logging.info("Connected to the BEST antenna server")
        """
        ANTENNA_IP = "192.168.30.134"  # ETH-RS232 Adapter IP connected to COM2 of Antenna
        ANTENNA_IP = "127.0.0.1"
        ANTENNA_PORT = 5002 
        antenna = Pointing(ip=ANTENNA_IP, port=ANTENNA_PORT)
        try:
            if not antenna == None and antenna.connected:
	            print("Antenna connection established!")
            else:
                raise IndexError()
            # print '>', antenna.get_status_string()
            # print antenna.get_dec()


        except IndexError:
            print "Index Error quitting"
            self.stop_best2_server()
            exit()        
        """
        self.current_pointing =  self.get_current_declination()
        logging.info("Current pointing is {} DEG".format(self.current_pointing))
        # print 'ccurnrrent', self.current_pointing
        # Keep track of current pointing
        # self.current_pointing = self.get_current_declination()

    def _launch_new_server(self):
        logging.warning("BEST antenna server is not available. Launching new BEST antenna server")
        self._thread = threading.Thread(target=self._start_best2_server_worker, name='BEST Server')
        self._thread.start()

        time.sleep(1)

    def _start_best2_server_worker(self):
        """
        Start the BEST-II server (to be run within a separate thread)

        :return:
        """

        self._best2_server = SdebTCPServer(("", self._port))
        try:
            while not self._stop_server:
                self._best2_server.handle_request()
        except socket.error:
            logging.exception('Socket Error in BEST-II server. Could not handle request.')
        except AttributeError:
            logging.warning('Could not handle request. BEST server offline.')
        finally:
            self._stop_server = False
            logging.info("BEST-II worker thread stopped")

    def stop_best2_server(self):
        """
        Stop the BEST-II server

        :return:
        """

        if self._connected:
            logging.info("Stopping BEST-II server")
            self._socket.close()
            if self._best2_server is not None:
                self._stop_server = True
                # while self._stop_server:
                #     logging.info("Waiting for BEST-II server to stop")
                #     time.sleep(1)

                self._best2_server.server_close()
                del self._best2_server
                self._best2_server = None
            self._connected = False

            logging.info('BEST-II server stopped')
        else:
            logging.info('Could not stop BEST server as it is not connected.')

    def get_current_declination(self):
        """
        Get current BEST2 declination

        :return:
        """

        if not self._connected:
            raise BEST2PointingException(
                "Could not retrieve current declination. BEST2 antenna server is not connected")

       
        data = None
        for _ in range(3):
            time.sleep(1)

            self._socket_send("best")

            # Wait for reply
            time.sleep(2)

            data = self._socket_recv()

            if re.search("[0-9]+", data) is not None:
                break

        # Ensure that the returned declination is a float        
        if re.search("[0-9]+", data) is None:
            raise BEST2PointingException("BEST: Could not get current declination (got `%s`)" % data)

        try:
            if data.startswith("ONSOURCE"):
                data = re.findall("[-+]?[0-9]*\.?[0-9]+", data)[0]

            # print repr(data)
            self.current_pointing = float(data)
        except IndexError:
            raise BEST2PointingException("BEST: Could not get current declination (got `%s`)" % data)

        return self.current_pointing

    def move_to_zenith(self):
        """
        Move BEST2 to zenith, which is 44.52

        :return:
        """

        return self.move_to_declination(self._zenith)

    def _socket_send(self, cmd):
        try:
            self._socket.sendall(cmd)
        except socket.timeout:
            raise BEST2PointingException("BEST Server: Socket timeout on command `{}`".format(cmd))
        except socket.error:
            raise BEST2PointingException("BEST Server: Socket error on command `{}`".format(cmd))

    def _socket_recv(self):
        try:
            return self._socket.recv(self._buffer_size)
        except socket.timeout:
            raise BEST2PointingException("BEST Server: Socket timeout on recv")
        except socket.error:
            raise BEST2PointingException("BEST Server: Socket error on recv")

    def move_to_declination(self, dec):
        """
        Move BEST2 to a particular declination

        :param dec: Declination to move to
        :raises BEST2PointingException:
        :return:
        """

        if not self._connected:
            raise BEST2PointingException("Could not move to declination. BEST2 antenna server is not connected")

        # Check if declination is valid
        if not -3.0 <= dec <= 90:
            raise BEST2PointingException("BEST2: Declination %s is out of range. Range is -3 to 90" % dec)

        # Check if we need to move at all
        if abs(self.current_pointing - dec) < self.DEC_ERROR_THOLD:
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

        raise BEST2PointingException("BEST2: Could not be pointed")

    def _move_best2(self, dec):
        """
        Issue the commands to move BEST2

        :param dec:
        :return:
        """
        # Issue command
        self._socket.sendall("best %.2f \r" % dec)
        time.sleep(2)
        data = self._socket.recv(self._buffer_size)

        logging.info("BEST2: Pointing - %s" % data)

        # Wait for start delay
        time.sleep(START_DELAY)

        tries = 0

        # Wait until pointing is very close to desired one
        while not self._stop_server:
            self._socket.sendall("progress")
            data = self._socket.recv(self._buffer_size)
            
            # parsed_msg = data.split("   ")
            try:               
                parsed_msg = re.findall("[-+]?[0-9]*\.?[0-9]+", data)

                if isinstance(parsed_msg, (list,)):                     
                    current_dec = float(parsed_msg[1])
            except IndexError:
                logging.warning('BEST2: Server returned: {}'.format(data))

                # Wait for a while before re-trying the command
                time.sleep(1)

                # Try the command N times, before breaking the loop.
                if tries > 3:
                    logging.warning('BEST2: Failed to move the Antenna. Skipping pointing.')
                    break

                tries += 1
            else:
                self.current_pointing = current_dec
                
                logging.info("Current pointing: {:0.2f}".format(self.current_pointing))
                
                if abs(current_dec - dec) < self.DEC_ERROR_THOLD:
                    logging.info("Antenna in position. DEC: {:0.2f}".format(current_dec))          
                    
                    return True

                time.sleep(2)    


        # Check if pointing was successful
        # curr_declination = self.get_current_declination()
        # if type(curr_declination) is not float:
        #     logging.warning("BEST2: Could not validate BEST2 movement")
        #     return False

        # if abs(curr_declination - dec) > self.DEC_ERROR_THOLD:
        #     print curr_declination ,  dec ,self.DEC_ERROR_THOLD
        #     logging.warning("BEST2: Failed to reach requested declination of DEC: {:0.2f}".format(dec))

        #     return False

        return False


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

    # best2 = BEST2.Instance()

    # best2.connect()

    # best2.stop_best2_server()

    best2 = BEST2.Instance()

    best2.connect()

    best2.move_to_declination(40)
    time.sleep(2)
    best2.move_to_declination(46)

    # best2.move_to_declination(42)

    best2.stop_best2_server()
