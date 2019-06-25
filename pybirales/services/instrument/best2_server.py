#!/usr/bin/env python

# Copyright (C) 2016, Osservatorio di RadioAstronomia, INAF, Italy.
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
# 
# Correspondence concerning this software should be addressed as follows:
#
#	Andrea Mattana
#	Radiotelescopi di Medicina
#       INAF - ORA
#	via di Fiorentina 3513
#	40059, Medicina (BO), Italy

"""
Provides a socket comunication between the RS-232 lane 
of the Northern Cross Pointing System Computer 
done by Andrea Maccaferri for the Pulsar Scheduler(1990)
"""

# Python modules
import SocketServer
import struct
import socket
import time

import datetime
import logging
import logging.handlers
import os, sys

import select

__author__ = "Andrea Mattana"
__copyright__ = "Copyright 2016, Osservatorio di RadioAstronomia, INAF, Italy"
__credits__ = ["Andrea Mattana, Andrea Maccaferri"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Andrea Mattana"

# Handy aliases
BUFF_LEN = 1024

# Preferred Pointing
STOW = 44.7

# CMDs
ASK_STATUS = "STA\r"
CHK_STATUS = 'CHK\r'
MOVE_BEST = 'NS2 %s \r'
MOVE_GO = 'GO \r'

# ANT_NUM
EW = 0
NORD2 = 1
NORD1 = 2
SUD1 = 3
SUD2 = 4

# ERROR CODES
ERR01 = "\tETH-RS232 Connection failed!"
ERR02 = "\tETH-RS232 CMD echo is not equal to the CMD sent"


class Pointing:
    def __init__(self, ip="192.168.30.134", port=5002):
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.settimeout(10)
        self.connected = False
        try:
            self._conn.connect((ip, port))
            self.connected = True
        except:
            logging.error(ERR01)
            return None
        # self._conn.settimeout(1)
        self.waiting = 1
        self.moving = 0
        self.buff_len = BUFF_LEN

    def _readLine(self):
        line = ''
        cr = ''
        while not cr == '\r':
            cr = self._conn.recv(1)
            line = line + cr
        return line[1:]

    def _send_cmd(self, cmd):
        self._conn.send(cmd)
        time.sleep(0.2)
        cmd_echo = self._readLine()
        if cmd_echo == cmd:
            return True
        else:
            return False

    def get_dec(self):
        self._conn.send(ASK_STATUS)
        time.sleep(0.5)
        _ = self._readLine()
        
        self.waiting = 0
        while not self.waiting:
            try:
                ans = self._conn.recv(BUFF_LEN)
                self.waiting = 1
            except:
                time.sleep(0.1)
        return ans.split()[NORD1]

    def set_dec(self, newdec):
        self._conn.send(MOVE_BEST % newdec)
        time.sleep(1)
        self.waiting = 0
        while not self.waiting:
            try:
                ans = self._conn.recv(BUFF_LEN)
                self.waiting = 1
            except:
                time.sleep(0.1)
        return ans

    def move_go(self):
        try:
            self._conn.send(MOVE_GO)
            self.moving = 1
            time.sleep(0.1)
        except:
            time.sleep(0.1)
        return "MOVING!"

    def check(self):
        try:
            if self._send_cmd(CHK_STATUS):
                ans = self._readLine()
                return ans
            else:
                return ERR02
        except:
            return ERR01

    def get_status_string(self):
        try:
            if self._send_cmd(ASK_STATUS):
                ans = self._readLine()
                # print ans
                return ans
            else:
                return ERR02
        except:
            return ERR01

    def close(self):
        self._conn.close()
        return


class SdebTCPHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        while True:
            self.data = self.request.recv(1024)
            if not self.data:
                time.sleep(0.1)
                self.request.close()
                break
            args = self.data.split()
            try:
                res = self.server.execute(args)
                self.request.send(res)
            except:
                time.sleep(1)


class SdebTCPServer(SocketServer.TCPServer):
    def __init__(self, addr):
        try:
            self.timeout = 1
            SocketServer.TCPServer.__init__(self, addr, SdebTCPHandler)
        except:
            pass
        self.rec = 0
        self.commands = {
            "status": self.status,
            "check": self.check,
            "best": self.best,
            "abort": self.abort,
            "progress": self.progress
        }

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

    def progress(self):
        try:
            P = Pointing()
            try:
                ans = P._readLine()
            except:
                return "ERROR - Pointing Computer might be SWITCHED OFF!!"
        except:
            return "ERROR - Can\'t be estabilished a connection with the ETH-RS232 adapter!"
        P.close()
        del P
        return ans

    def status(self):
        try:
            P = Pointing()
            try:
                ans = P.get_status_string()
            except:
                return "ERROR - Pointing Computer might be SWITCHED OFF!!"
        except:
            return "ERROR - Can\'t be estabilished a connection with the ETH-RS232 adapter!"
        P.close()
        del P
        return ans

    def best(self, newdec='dont_move'):
        try:
            P = Pointing()
            try:
                if newdec == "dont_move":
                    ans = P.get_status_string().split()[NORD1]
                else:
                    ans = P.get_status_string()
                    P.set_dec(newdec)
                    P.move_go()
                    ans = "MOVING!"

            except Exception as e:
                return "ERROR - Pointing Computer might be SWITCHED OFF!!"
        except:
            return "ERROR - Can\'t be estabilished a connection with the ETH-RS232 adapter!"
        P.close()
        del P
        return ans

    def check(self):
        try:
            P = Pointing()
            try:
                ans = P.check()
                if ans[6:8] == "98":
                    ans = "OK"
                elif ans[6:8] == "66":
                    ans = "WARN LOCAL"
            except:
                return "ERROR - Pointing Computer might be SWITCHED OFF!!"
        except:
            return "ERROR - Can\'t be estabilished a connection with the ETH-RS232 adapter!"
        P.close()
        del (P)
        return ans

    def abort(self):
        self.rec = 0
        return "BEST-II server received aborted call"

    def execute(self, args):
        if not self.commands.has_key(args[0]):
            res = "Command \'%s\' not found." % (args[0],)
            logging.error(res)
        else:
            if len(args) > 1:
                try:
                    res = self.commands[args[0]](*(args[1:]))
                except TypeError, te:
                    time.sleep(0.1)
            else:
                try:
                    res = self.commands[args[0]]()
                except TypeError, te:
                    time.sleep(0.1)
        return res