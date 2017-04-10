# BIRALES processing backend

## Installation

Clone repository
```
git clone https://xxx@bitbucket.org/lessju/birales.git
```

Install PyBirales
```
cd python/
python setup.py install 
```


## Dependencies
* Python 2.7
* MongoDB


## Usage
List available pipelines
```
birales.py --help
```

Help on how to run a given [Pipeline Name] 
```
birales.py [Pipeline Name] --help
```

Running a [Pipeline Name] 
```
birales.py [Pipeline Name] [OPTIONS] CONFIGURATION
```

Options:
 * The `--debug / --no-debug` option specifies whether (or not) you'd like to log debug messages.

Arguments:
 * `CONFIGURATION` The path of the configuration file that the pipeline should use.     


## Post-processing
Start the Flask server on port 5000
```
python birales.py start_server --port=5000 --debug
```

*Deprecated:* Post-Process a data set XXX in observation YYY (32 beams)
```
python post.py post_process --observation YYY --data_set XXX --n_beams=32
```

*Deprecated:* Reset the database of data set XXX in observation YYY
```
python post.py reset --observation YYY --data_set XXX
```