from flask import Flask, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model
import pickle
import gzip
import joblib

app = Flask(__name__)

def decompress_pickle_gzip(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

model = decompress_pickle_gzip('assets/model.pkl.gz')
model = joblib.load(model)

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
        print(json_data)
        if not json_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Convert JSON data to DataFrame
        try:
            if isinstance(json_data, list):
                # Handle list of JSON objects
                json_to_df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Handle single JSON object
                json_to_df = pd.DataFrame([json_data])
            else:
                return jsonify({'error': 'Unsupported JSON format'}), 400

            # Ensure required columns are present
            required_columns = ['hum', 'temp', 'pm2_5']
            for col in required_columns:
                if col not in json_to_df.columns:
                    return jsonify({'error': f'Missing column: {col}'}), 400

        except ValueError as e:
            return jsonify({'error': f'Error creating DataFrame: {str(e)}'}), 400

        # Apply the formula to calculate Corrected PM2_5
        try:
            predictions_array = model.predict(json_to_df)
            predictions_df = pd.DataFrame(predictions_array, columns=['calibrated_pm2_5'], index = json_to_df.index)

        except Exception as e:
            return jsonify({'error': f'Error calculating calibrated PM2_5: {str(e)}'}), 500

        return predictions_df.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
      
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)

