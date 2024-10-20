from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the preprocessor and model from the pickle files
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def welcome():
    return render_template('front.html')

# Route to render HTML form
@app.route('/form')
def home():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict_datapoint', methods=['POST'])
def predict_datapoint():
    # Get form data
    carat = float(request.form['carat'])
    depth = float(request.form['depth'])
    table = float(request.form['table'])
    cut = request.form['cut']
    color = request.form['color']
    clarity = request.form['clarity']

    # Create a DataFrame from user input
    input_data = pd.DataFrame([[carat, depth, table, cut, color, clarity]], 
                              columns=['carat', 'depth', 'table', 'cut', 'color', 'clarity'])

    # Preprocess the input data using the saved preprocessor
    processed_data = preprocessor.transform(input_data)

    # Make the prediction using the saved model
    prediction = model.predict(processed_data)

    # Render the result page and pass the prediction value
    return render_template('result.html', price=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
