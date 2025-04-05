import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare your data
df = pd.read_excel("updated_diabetes_food_recommendation.xlsx")  # Update the file path if necessary
x = df[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']]
y = df[['Diabetes Level']]
l = LabelEncoder()
y['Diabetes Level'] = l.fit_transform(y['Diabetes Level'])
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=2)
rf.fit(x_scaled, y)

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()  # Expecting JSON data in the request
    glucose = data['glucose']
    blood_pressure = data['bloodPressure']
    insulin = data['insulin']
    bmi = data['bmi']
    age = data['age']
    
    # Prepare input for prediction
    input_data = np.array([[glucose, blood_pressure, insulin, bmi, age]])
    input_scaled = sc.transform(input_data)

    # Predict diabetes level
    prediction = rf.predict(input_scaled)
    diabetes_level = l.inverse_transform(prediction.astype(int))[0]

    # Get food recommendation based on diabetes level
    food_recommendation = df.loc[df['Diabetes Level'] == diabetes_level, 'Food Recommendation'].values

    # Prepare response
    response = {
        'diabetes_level': diabetes_level,
        'food_recommendation': food_recommendation.tolist()  # Convert to list if necessary
    }

    return jsonify(response)  # Return the response in JSON format

if __name__ == '__main__':
    app.run(debug=True)