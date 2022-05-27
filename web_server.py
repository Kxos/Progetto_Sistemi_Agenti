from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from test import evaluate
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='http://localhost')

log = logging.getLogger('werkzeug')
log.disabled = True


@socketio.on('message')
def handle_message(data):
    print('received message: ', data.get('data'))

    # messaggio_di_ritorno = {
    #     "emotion_class": 0,
    #     "Valenza": 0.246785346,
    #     "Arousal": 1.675243543,
    # }
    messaggio_di_ritorno = evaluate(data.get('data'))

    if messaggio_di_ritorno is not None:
        send(messaggio_di_ritorno)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    socketio.run(app, host="localhost")

    # TODO - Caricamento dei 3 modelli
