from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor


app = Flask(__name__)


@app.route("/")
@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            predict_dict = request.form.to_dict()
            predict_list = list(map(float, list(predict_dict.values())))
            predict_list = np.asarray(predict_list)
            list_predicted = []
            model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
            list_predicted.append(model.predict(predict_list.reshape(1, -1))[0])
            model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
            list_predicted.append(model.predict(predict_list.reshape(1, -1))[0])
            model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
            list_predicted.append(model.predict(predict_list.reshape(1, -1))[0])
            model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
            list_predicted.append(model.predict(predict_list.reshape(1, -1))[0])
            model = pickle.load(open('models/Random_Forest.pkl', 'rb'))
            list_predicted.append(model.predict(predict_list.reshape(1, -1))[0])
    except:
        return render_template("diabetes.html", message='Invalid Data')
    return render_template('predict.html', pred=list_predicted)


if __name__ == '__main__':
    app.run(debug = True)