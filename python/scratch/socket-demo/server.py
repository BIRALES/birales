import socketio
import eventlet.wsgi
import time
import pickle
import numpy as np
import json

sio = socketio.Server()
app = Flask(__name__)


@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')


@sio.on('connect')
def connect(sid, environ):
    print("connected", sid)


@sio.on('disconnect', namespace='/chat')
def disconnect(sid):
    print('disconnect ', sid)


@sio.on('data')
def get_data(sid):
    a = np.ones((1, 32, 1000, 64))
    i = 0
    while i < 5:
        i += 1
        sio.emit('event', json.dumps(a.tolist()))
        time.sleep(5)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8000)), app)
