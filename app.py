from flask import Flask, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

app = Flask(__name__)

# Load the model once, when the app starts
model = load_model('assets/correction_factor_random_forest_sensor960-25-April-2024')

@app.route('/')
def index():
    return 'Welcome to the Calibration Engine API!'

@app.route('/calibration-engine-api/PM25/v1/', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return jsonify({'Instruction': 'Send JSON data with Hum, Temp, and PM2_5 for calibration'})

    # POST: Handle calibration prediction
    try:
        # Ensure there is JSON data and it is the correct format
        json_data = request.json
        if not json_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Create DataFrame from JSON, handle potential errors in key names
        try:
            json_to_df = pd.read_json(json_data)
            df_for_calibration = json_to_df[['Hum', 'Temp', 'PM2_5']]
        except KeyError as e:
            return jsonify({'error': f'Missing columns in the data: {str(e)}'}), 400

        # Make predictions
        predictions = predict_model(model, data=df_for_calibration)
        predictions.rename(columns={'prediction_label': 'calibrated_PM2_5'}, inplace=True)

        # Combine predictions with original data
        combined_df = pd.concat([json_to_df, predictions['calibrated_PM2_5']], axis=1)
        
        return combined_df.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)

