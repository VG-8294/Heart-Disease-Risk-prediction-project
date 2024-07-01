#This Python code is a Flask web application that serves as an API for predicting heart disease risk based on user inputs.


from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import pickle

app = Flask(__name__) #flask instance created
CORS(app) #corse imported/enabled

# Load the pickled model
with open('random_forest_model.pkl', 'rb') as file: #pickle is used to load trained mchine learning model
    model = pickle.load(file)

# Define the input features
features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
            'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',
            'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime']

# Function to predict heart disease based on user inputs
def predict_heart_disease(user_inputs):
    input_data = np.array([[user_inputs[feature] for feature in features]]) #numpy library is used to covert input into 2D array
    print(input_data)
    prediction = model.predict(input_data) #model.predict used to predict 2D array created through numpy
    return prediction

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request body
    input_data = request.json
    print("Input data:", input_data)
    
    # Check if input data is provided
    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400
    
    try:
        # Predict heart disease
        prediction = predict_heart_disease(input_data)
        print("Prediction:", prediction)
        
        
        # Output prediction result
        # if prediction == 1:
        #     result = "You are predicted to have heart disease."
        # else:
        #     result = "You are predicted not to have heart disease."
        
        return jsonify({'prediction': int(prediction)}) #jsonify is used to convert output of API to json object
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

if __name__ == '__main__':
    app.run(debug=True)
#python3 -m venv venv
#source venv/bin/activate
#pip install flask
#pip install scikit-learn==1.2.2
#pip install flask-cors
#python script.py