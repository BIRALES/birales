# BIRALES processing backend

## Dependencies
* Python 2.7
* MongoDB

## Installation

The BIRALES space debris detection system was built in Python 2.7 for the Ubuntu 16.04 operating system. Data is persisted to a MongoDB v3.2 database. The installation procedure for the system are detailed below:

The MongoDB database can be installed through:

```
sudo apt-key adv —keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
echo “deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo mongod start 
```

Clone the git repository using:
```
git clone https://bitbucket.org/lessju/birales.git
```

Then, the PyBirales python application can then be installed:
```
python setup.py install
```

Once the dependencies have been installed, the beamformer module can be built.
```
cd src/
mkdir build
cd build
cmake ..
sudo  make install
```

This completes the procedure needed to install the PyBirales application.

## Usage
List available commands
```
birales --help
```

### Pipelines

Help on how to run a given pipeline (e.g. the Detection Pipeline)
```
birales pipelines [Pipeline Name] --help
```

Running a pipeline
```
birales pipelines [Pipeline Name] [OPTIONS] CONFIGURATION
```

Options:
 * The `--debug / --no-debug` option specifies whether (or not) you'd like to log debug messages.
 * The `--duration` how long the pipeline will run

Arguments:
 * `CONFIGURATION` The path of the configuration file that the pipeline should use.      

### Scheduler
Run the scheduler using a specified JSON config file
```
birales schduler -s [SCHEDULER CONFIGURATION FILE]
```

### Calibration
Run the calibration routine 
```
birales services calibration [OPTIONS] CONFIGURATION
```

### Front-end
Start the Flask server on port 5000
```
birales-frontend start_server --port=5000 --debug
```

Front-end application will be served on http://127.0.0.1
