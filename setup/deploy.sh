#!/usr/bin/env bash
# Clone repository from BitBucket

# Install system requirements
sudo apt update
sudo apt install python3-pip
sudo apt install redis
sudo apt install python3-pip

# Create and load virtualenv
virtualenv -p python3 venv
source venv/bin/activate

# Go to Python directory
cd birales/python

# Install PyFABIL
pip install git+https://gitlab.com/ska-telescope/pyfabil.git

# Install BIRALES python package and requirements
pip install .
pip install -r requirements_birales.txt

# For the frontend, install the required packages and define environmental variables
sudo apt install npm
pip install -r requirements_frontend.txt
cd python/frontend/static
npm install
export BIRALES__DB_USERNAME=birales_rw
export BIRALES__DB_PASSWORD=birales

# Optional:
# - Slack Webclient needs to be installed for slack
# - Cupy needs to be installed for GPU




