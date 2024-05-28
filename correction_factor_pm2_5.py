from pycaret.regression import *
import pandas as pd

import pickle
import gzip

def compress_pickle_gzip(file_path, data):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
def main(df):
    s = setup(df, target = 'pm2_5_ref', session_id = 123)

    # compare baseline models
    best1 = compare_models()

    # predict on test set
    holdout_pred1 = predict_model(best1)

    save_model(best1, 'assets/correction_factor_random_forest_sensor960-pm2_5-28-May-2024')

    # Usage
    model = 'assets/correction_factor_random_forest_sensor960-pm2_5-28-May-2024.pkl'  
    file_path = 'assets/model_pm2_5.pkl.gz'  
    compress_pickle_gzip(file_path, model)

if __name__ == "__main__":
    df = pd.read_csv('assets/merged.csv', parse_dates=["DataDate"])

    filtered_df1 = df[["pm2_5_ref", "pm2_5", "temp", "hum"]]
    print(filtered_df1)
    main(filtered_df1)
