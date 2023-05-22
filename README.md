# BIRALES processing backend

## Dependencies
* Python 3+
* MongoDB
* NPM 3+
* REDIS

## Installation

The BIRALES space debris detection system was built in Python 3+ for the Ubuntu 16.04 operating system.
Data is persisted to a MongoDB v3.2 database whilst a REDIS database is used as a message broker.
The installation procedure for the system are detailed below:

### Database
The MongoDB database can be installed through:

```bash
sudo apt-key adv �keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
echo �deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo mongod start 
```

### Download Source Files
Clone the git repository using:
```bash
git clone https://bitbucket.org/lessju/birales.git
```

Then, the PyBirales python application can then be installed:
```bash
pip install -r requirements.txt
python setup.py install
```

Once the dependencies have been installed, the beamformer module can be built.
```bash
cd  pybirales/pipeline/modules/beamformer/src/
mkdir build
cd build
cmake ..
sudo  make install
```

The front-end dependencies can be installed through NPM

```bash
cd pybirales/app/static/
npm install
```

This completes the procedure needed to install the PyBirales application.

---

## Usage

The BIRALES system

The help message for the BIRALES system can be accessed through,

```bash
birales --help
```

The BIRALES system can be split into three main components

1. Pipelines
2. Scheduler
3. Services

These components need to be configured accordingly using `.ini` configuration files. Configuration templates are provided in the BIRALES application data directory.

##### The BIRALES application data

When the BIRALES system is installed, a `.birales` directory is installed in the user's home directory. The purpose of this directory is to store the application data of the system and is not intended to be versioned. The user can find the following sub-directories:

- `~/.birales/visualisation` 
  - Visualisation (e.g. FITS) files which are often useful for debugging purposes
- `~/.birales/configuration` 
  - Template configuration files for the BIRALES system for development and production systems
- `~/.birales/schedules` 
  - Template files for BIRALES scheduler

### Running a Pipeline

Currently, the following pipelines are implemented: 

- Correlation Pipeline `correlation_pipeline`
- Detection Pipeline `detection_pipeline`
- Stand Alone Pipeline `standalone_pipeline`

The help message for a pipeline can be invoked through,

```bash
birales pipelines [Pipeline Name] --help
```

Whilst the pipeline can be run through
```bash
birales pipelines [Pipeline Name] [OPTIONS]
```

The following options are available:

* The `-c / --config` option specifies the BIRALES configuration file to use. If multiple configuration options are
  specified, they will override each other. Required.
* The `--debug / --no-debug` option specifies whether (or not) you'd like to log debug messages.
* The `--duration` option specifies how long the pipeline will run.
* The `--pointing` option specifies the declination of the BEST-II.
* The `--tx ` option specifies t;he transmission frequency in MHz.

#### Example

```bash
birales pipelines msds-detection-pipeline -c ~/.birales/configuration/birales.ini -c ~/.birales/configuration/detection.ini
```

> **_NOTE:_**  In this case, the configuration parameters in _detection.ini_ will override those in _birales.ini_

### Running the Scheduler

Run the scheduler using a specified JSON config file

```bash
birales scheduler [OPTIONS]
```

The following options are available:

- The `-c / --config` option specifies the BIRALES configuration file to use. If multiple configuration options are
  specified, they will override each other. Required.
- The `-f` option specifies the format of the schedule file (*tdm* or *json*).
- The `-s` option specifies the schedule that the scheduler will follow.

#### Example

```bash
birales scheduler -c ~/.birales/configuration/birales.ini
```

### Running the Calibration Service

Run the calibration routine

```bash
birales services calibration [OPTIONS]
```

The following options are available:

- The `-c / --config` option specifies the BIRALES configuration file to use. If multiple configuration options are specified, they will override each other. Required.

The calibration coefficients can be reset (0 for amplitude and 1 for gain) through:

```bash
birales services reset_coefficients -c [CONFIGURATION]
```

### Running the BIRALES Web application

Start the Flask server (still in development)
```bash
python pybirales/app/app.py 
```

Front-end application will be served on http://127.0.0.1
