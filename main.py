from flask import Flask, request
from os import listdir
from os.path import join
from keras.models import load_model

from flask_cors import CORS, cross_origin

from numpy import asarray
import json

app = Flask(__name__)
CORS(app)

# LOADING KERAS MODELS
MODEL = {}
MODEL_DIR = './models'
for f in listdir(MODEL_DIR):
    #if f == 'because.model':
    fpath = join(MODEL_DIR, f)
    word = f.split('.')[0]
    MODEL[word] = load_model(fpath)
    #MODEL[word]._make_predict_function()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/pronserv', methods=['POST'])
def pron_serv():
    data = request.get_json()
    feats = data['feats']
    w = data['word']
    print feats, w
    words = w.split()
    pred = None
    for ww in words:
        ww = ww.strip()
        if ww:
            pred = MODEL[ww].predict(asarray(feats).reshape(1, -1))[0][1]
    print pred
    return json.dumps({'pred': str(round(pred, 2))})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, threaded=False)
