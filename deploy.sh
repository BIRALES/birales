# Check that the dependencies are met. If not install.

# Deploy the BIRALES application
python setup.py install

# Compile the beamformer library
cd pybirales/pipeline/beamformer/src
mkdir build
cd build
cmake ..
sudo  make install

# Create the logging directory
mkdir /var/log/birales
chown -R lessju:lessji /var/log/birales

# Change user permission for the BIRALES home folder
chown -R lessju:lessji ~/.birales