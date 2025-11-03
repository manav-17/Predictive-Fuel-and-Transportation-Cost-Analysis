#Importing the Libraries
import pandas as pd  #Import pandas for creating and manipulating DataFrames.
import joblib #Import joblib to load your pre-trained machine learning models.
import requests #Import requests to make live API calls (for weather).
import warnings #Import warnings to suppress unnecessary warning messages in the console.
from datetime import datetime #Import datetime for handling date objects (though it's not explicitly used).
from flask import Flask, request, jsonify #Import Flask to create the web server and API endpoints.
from flask_cors import CORS #Import CORS to allow your web page (on a different origin)
warnings.filterwarnings('ignore', category=UserWarning) #Suppress UserWarning

app = Flask(__name__) # Initialize the Flask application.
CORS(app)

#Import os to handle file paths
import os
model_cost = joblib.load(os.path.join(os.path.dirname(__file__), 'Regression_model.pkl')) #Load the trained Regression model
model_risk = joblib.load(os.path.join(os.path.dirname(__file__),'Classification-model.pkl')) #Load the trained Classification model
import os
fuel_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'fuel_prices.csv')) #Load the historical fuel price data
fuel_df['Date'] = pd.to_datetime(fuel_df['Date']) #Convert the Date column to datetime.

#Define the coordinates for cities.
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
        exact_match = city_data[city_data['Date'] == user_date] #Try to find an exact match for the date.

        if not exact_match.empty:
            row = exact_match.iloc[0]
            print(f"Fuel Price: Found exact match for {date_str}.")
            return row['Petrol_Price'], row['Diesel_Price'], row['Date']
        else:
            fallback_data = city_data[city_data['Date'] <= user_date] #If no exact match, find the most recent price before or on the user's date.
            if not fallback_data.empty:
                row = fallback_data.iloc[-1]
                print(f"Fuel Price: No exact match. Using most recent from {row['Date'].strftime('%Y-%m-%d')}.")
                return row['Petrol_Price'], row['Diesel_Price'], row['Date']
            else:
                row = city_data.iloc[0] #If the user's date is before any data we have, use the oldest available price.
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
        'start_date': date_str, #data for single day
        'end_date': date_str,
        'timezone': 'auto'
    }
    #Make the live API call.
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    results = response.json()
    
    #Extract the data from the JSON response.
    temp = results['daily']['temperature_2m_mean'][0]
    precip = results['daily']['precipitation_sum'][0]
    print(f"Weather: Fetched forecast for {date_str}: Temp {temp}C, Precip {precip}mm")
    return temp, precip


#Prediction 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json #Get the JSON data sent by the user's browser.
        print(f"Received request: {data}")

        user_date = data['date']
        user_city = data['city']
        user_vehicle = data['vehicle']
        user_route = data['route']
        user_distance = float(data['distance'])
        user_load = float(data['load'])
        
        #Fetch External Data
        petrol_price, diesel_price, _ = get_fuel_prices(user_city, user_date)
        temp, precip = get_weather_forecast(user_city, user_date)
        
        #Preprocess Data for Model
        fuel_type = 'Diesel' if user_vehicle in ['Truck', 'Van'] else 'Petrol'
        
        #Create a dictionary of the input features.
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
        
        #Define the exact column
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



