Commands via termina to power-on and power-off TPMs and monitor status:
  -  Connect to subrack via ssh: `sshpass -p SkaUser ssh -o StrictHostKeyChecking=no mnguser@subrack-0-cpu`
  -  Stop web_server: `sudo systemctl stop web_server`
  -  Control subrack: `ipython -i -- tools/subrack_monitor.py --skip_init`
  -  Monitor subrack: `python tools/subrack_monitor.py -s`