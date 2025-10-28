import pandas as pd
import joblib
import requests
import warnings
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
warnings.filterwarnings('ignore', category=UserWarning)
app = Flask(__name__)
CORS(app)

import os
model_cost = joblib.load(os.path.join(os.path.dirname(__file__), 'Regression_model.pkl'))
model_risk = joblib.load(os.path.join(os.path.dirname(__file__),'Classification-model.pkl'))
import os
fuel_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'fuel_prices.csv'))
fuel_df['Date'] = pd.to_datetime(fuel_df['Date'])

CITY_COORDS = {
    'Delhi': {'lat': 28.61, 'lon': 77.21},
    'Mumbai': {'lat': 19.08, 'lon': 72.88},
    'Chennai': {'lat': 13.08, 'lon': 80.27},
    'Kolkata': {'lat': 22.57, 'lon': 88.36}
}


def get_fuel_prices(city, date_str):
    try:
        user_date = pd.to_datetime(date_str)
        city_data = fuel_df[fuel_df['City'] == city].sort_values(by='Date')

        if city_data.empty:
            raise Exception(f"No fuel price data for city: {city}")
        exact_match = city_data[city_data['Date'] == user_date]

        if not exact_match.empty:
            row = exact_match.iloc[0]
            print(f"Fuel Price: Found exact match for {date_str}.")
            return row['Petrol_Price'], row['Diesel_Price'], row['Date']
        else:
            fallback_data = city_data[city_data['Date'] <= user_date]
            if not fallback_data.empty:
                row = fallback_data.iloc[-1]
                print(f"Fuel Price: No exact match. Using most recent from {row['Date'].strftime('%Y-%m-%d')}.")
                return row['Petrol_Price'], row['Diesel_Price'], row['Date']
            else:
                row = city_data.iloc[0]
                print(f"Fuel Price: Date is older than data. Using oldest from {row['Date'].strftime('%Y-%m-%d')}.")
                return row['Petrol_Price'], row['Diesel_Price'], row['Date']
    except Exception as e:
        print(f"Error in get_fuel_prices: {e}")
        raise


def get_weather_forecast(city, date_str):
    if city not in CITY_COORDS:
        raise Exception(f"No coordinates for city: {city}")

    coords = CITY_COORDS[city]
    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': coords['lat'],
        'longitude': coords['lon'],
        'daily': 'temperature_2m_mean,precipitation_sum',
        'start_date': date_str,
        'end_date': date_str,
        'timezone': 'auto'
    }

    response = requests.get(api_url, params=params)
    response.raise_for_status()
    results = response.json()

    temp = results['daily']['temperature_2m_mean'][0]
    precip = results['daily']['precipitation_sum'][0]
    print(f"Weather: Fetched forecast for {date_str}: Temp {temp}C, Precip {precip}mm")
    return temp, precip



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received request: {data}")

        user_date = data['date']
        user_city = data['city']
        user_vehicle = data['vehicle']
        user_route = data['route']
        user_distance = float(data['distance'])
        user_load = float(data['load'])

        petrol_price, diesel_price, _ = get_fuel_prices(user_city, user_date)
        temp, precip = get_weather_forecast(user_city, user_date)

        fuel_type = 'Diesel' if user_vehicle in ['Truck', 'Van'] else 'Petrol'

        new_trip_data = {
            'City': user_city,
            'Vehicle_Type': user_vehicle,
            'Fuel_Type': fuel_type,
            'Route': user_route,
            'Distance_km': user_distance,
            'Load_Weight_kg': user_load,
            'Petrol_Price': petrol_price,
            'Diesel_Price': diesel_price,
            'Temp_Mean_C': temp,
            'Precipitation_mm': precip
        }

        columns = [
            'City', 'Vehicle_Type', 'Fuel_Type', 'Route', 'Distance_km',
            'Load_Weight_kg', 'Petrol_Price', 'Diesel_Price', 'Temp_Mean_C',
            'Precipitation_mm'
        ]
        new_trip_df = pd.DataFrame([new_trip_data], columns=columns)

        predicted_cost = model_cost.predict(new_trip_df)[0]
        prediction_risk = model_risk.predict(new_trip_df)[0]
        probabilities_risk = model_risk.predict_proba(new_trip_df)[0]
        prob_over_budget = probabilities_risk[1]

        return jsonify({
            'predicted_cost': round(predicted_cost, 2),
            'probability_over_budget': round(prob_over_budget * 100, 1),
            'fetched_temp': temp,
            'fetched_precip': precip
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 400
import os 
@app.route('/')
def index():
    return open(os.path.join(os.path.dirname(__file__), 'index.html')).read()


if __name__ == '__main__':
    app.run(debug=True, port=5001)



