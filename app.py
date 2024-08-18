from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the CSV data
data = pd.read_csv('Forest_fire.csv')
# Prepare the data for basic heuristics or analysis
X = data[['Oxygen', 'Temperature', 'Humidity']]
y = data['Fire Occurrence']

def simple_predict(oxygen, temperature, humidity):
    # Calculate mean and standard deviation for thresholding
    mean_values = X.mean()
    std_values = X.std()
    thresholds = mean_values + (std_values * 0.5)
    
    # Compute the probability based on how far the input values are from the mean
    prob_oxygen = min(1, (oxygen / thresholds['Oxygen']))
    prob_temperature = min(1, (temperature / thresholds['Temperature']))
    prob_humidity = min(1, (thresholds['Humidity'] / humidity))
    
    # Combine probabilities into a simple model (you may want to refine this)
    probability = (prob_oxygen + prob_temperature + prob_humidity) / 3
    # Convert combined probability into a fire occurrence probability
    return probability, probability > 0.5

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input features from the form
        oxygen = int(request.form['oxygen'])
        temperature = int(request.form['temperature'])
        humidity = int(request.form['humidity'])
        
        # Make a prediction using the simple heuristic
        probability, is_in_danger = simple_predict(oxygen, temperature, humidity)

        # Format the output
        output = '{0:.2f}'.format(probability)
        if is_in_danger:
            pred_text = f'Your Forest is in Danger. Probability of fire occurring is {output}.'
        else:
            pred_text = f'Your Forest is safe. Probability of fire occurring is {output}.'

    except Exception as e:
        pred_text = f"Error: {str(e)}"

    return render_template('index.html', pred=pred_text)

if __name__ == '__main__':
    app.run(debug=True)
