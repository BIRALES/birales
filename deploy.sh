#!/usr/bin/env bash
# Clone repository from BitBucket


# Check that the dependencies are met. If not install.

# Install PyDAQ
git clone git@bitbucket.org:aavslmc/aavs-daq.git
cd aavs-daq/src
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/lib ..
sudo make install

# Install TCPO calibration pipeline
# git fetch origin cleaning_jbxm
# git checkout cleaning_jbxm
# git pull cleaning_jbxm
# sudo ./deploy.sh

# Deploy the BIRALES application
python setup.py install

# Compile the beamformer library
cd pybirales/pipeline/modules/beamformer/src
mkdir build
cd build
cmake ..
sudo  make install

# Create the logging directory
mkdir /var/log/birales
chown -R lessju:lessji /var/log/birales
CAP_SYS_NICE
# Change user permission for the BIRALES home folder
chown -R lessju:lessji ~/.birales

# Set the linux capabilities
sudo setcap cap_net_raw,cap_ipc_lock,cap_kill,cap_sys_nice+ep /usr/bin/python2.7

