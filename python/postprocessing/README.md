# BIRALES Post-processing pipeline

## useful commands
Start the Flask server on port 9000
<code>python main.py run_dev_server --port 9000 <code>

Post-Process a data set XXX in observation YYY (32 beams)
<code> python main.py post_process --observation YYY --data_set XXX --n_beams=32 <code>

Reset the database
<code> python main.py reset <code>