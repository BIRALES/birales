from pybirales.instrument.best2 import BEST2
import logging
from sys import stdout

log = logging.getLogger('')
log.setLevel(logging.INFO)
str_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler(stdout)
ch.setFormatter(str_format)
log.addHandler(ch)

best2 = BEST2()
best2.move_to_declination(45)
