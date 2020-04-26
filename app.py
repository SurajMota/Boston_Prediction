import numpy as np
from flask import Flask, request,jsonify,render_template
from logging import FileHandler, WARNING
import pickle
import math

app = Flask(__name__)

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

app.logger.addHandler(file_handler)

model = pickle.load(open('HousePrediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('index.html')
    int_features =[float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text="MEDV or Price of house in 1000 Dollar will be  {}".format(output))

if __name__ == '__main__':
    app.run()
