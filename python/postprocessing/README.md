# BIRALES Post-processing pipeline

## Documentation
Start the Flask server on port 9000
`python main.py run_dev_server --port 9000`

Post-Process a data set XXX in observation YYY (32 beams)
`python main.py post_process --observation YYY --data_set XXX --n_beams=32`

Reset the database of data set XXX in observation YYY
`python main.py reset --observation YYY --data_set XXX`