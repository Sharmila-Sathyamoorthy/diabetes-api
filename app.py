import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# Load and prepare the CSV data
df = pd.read_csv("data.csv")
x = df[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']]
y = df[['Diabetes Level']]
l = LabelEncoder()
y['Diabetes Level'] = l.fit_transform(y['Diabetes Level'])
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=2)
rf.fit(x_scaled, y.values.ravel())

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    input_data = np.array([[data['glucose'], data['blood_pressure'], data['insulin'], data['bmi'], data['age']]])
    input_scaled = sc.transform(input_data)
    prediction = rf.predict(input_scaled)
    diabetes_level = l.inverse_transform(prediction.astype(int))[0]

    food_recommendation = df.loc[df['Diabetes Level'] == diabetes_level, 'Food Recommendation'].unique()

    return jsonify({
        'diabetes_level': diabetes_level,
        'food_recommendation': food_recommendation.tolist()
    })

if __name__ == '__main__':
    from os import environ
    port = int(environ.get("PORT", 5000))  # Render sets this dynamically
    app.run(host='0.0.0.0', port=port)

