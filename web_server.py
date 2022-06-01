from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from test import evaluateFrame, loadModels
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='http://localhost:5000')

log = logging.getLogger('werkzeug')
log.disabled = True

val_preprocess, device, model_emotion_class, model_Valenza, model_Arousal = loadModels()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    # print('received message: ', data.get('data'))

    # messaggio_di_ritorno = {
    #     "emotion_class": 0,
    #     "Valenza": 0.246785346,
    #     "Arousal": 1.675243543,
    # }

    messaggio_di_ritorno = evaluateFrame(model_emotion_class, model_Valenza, model_Arousal, data.get('data'), val_preprocess, device)

    if messaggio_di_ritorno is not None:
        send(messaggio_di_ritorno)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    socketio.run(app, host="localhost")
