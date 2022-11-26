from flask import Flask, request, render_template
from flask import Flask, render_template, request, redirect
import librosa
import soundfile
import os, glob, pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


# @app.route('/api', methods=['POST', 'GET'])
# def api_response():
#     from flask import jsonify
#     if request.method == 'POST':
#         return jsonify(**request.json)


@app.route('/', methods=['POST'])
def predict():
    audioFile = request.files['audioFile']
    audio_path = "./audios/" + audioFile.filename
    audioFile.save(audio_path)
    model = 'C:/Users/vikas/Desktop/InHouse/SER/rawdess/modelForPrediction1.sav'
    loaded_model = pickle.load(open(model, 'rb'))  # loading the model file from the storage
    feature = extract_feature(audio_path)
    feature = feature.reshape(1, -1)
    prediction = loaded_model.predict(feature)

    return render_template('index.html', p=prediction[0])
def extract_feature(file_name):
    mfcc = True
    chroma = True
    mel = True
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result
if __name__ == '__main__':
    app.run()
