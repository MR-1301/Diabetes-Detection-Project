from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble.forest import ForestClassifier, ForestRegressor


app = Flask(__name__)

def predict(values, dic):

    values = np.asarray(values)
    list=[];
    model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
    list.append(model.predict(values.reshape(1, -1))[0])
    model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
    list.append(model.predict(values.reshape(1, -1))[0])
    model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
    list.append(model.predict(values.reshape(1, -1))[0])
    model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
    list.append(model.predict(values.reshape(1, -1))[0])
    model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
    list.append(model.predict(values.reshape(1, -1))[0])
    return list

@app.route("/")
def home():
    return render_template('diabetes.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)

    return render_template('predict.html', pred = pred)


if __name__ == '__main__':
	app.run(debug = True)