# -*- coding: utf-8 -*-
#
# This file is part of the SKA Low MCCS project
#
# Distributed under the terms of the GPL license.
# See LICENSE.txt for more info.
"""This subpackage implements plugins for managing tile hardware."""

__author__ = "Alessio Magro"

# Helper to reduces import names

# Plugin Superclass
from pyfabil.plugins.firmwareblock import FirmwareBlock

# TPM plugins
from pybirales.digital_backend.plugins.tpm.tpm_debris_firmware import TpmDebrisFirmware

# TPM 1.6 plugins
from pybirales.digital_backend.plugins.tpm_1_6.tpm_debris_firmware import Tpm_1_6_DebrisFirmware
