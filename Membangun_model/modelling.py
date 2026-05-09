import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data():
    """Memuat data yang sudah diproses."""
    train_df = pd.read_csv('california_housing_preprocessing/train_processed.csv')
    test_df = pd.read_csv('california_housing_preprocessing/test_processed.csv')
    
    X_train = train_df.drop('median_house_value', axis=1)
    y_train = train_df['median_house_value']
    X_test = test_df.drop('median_house_value', axis=1)
    y_test = test_df['median_house_value']
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # 1. Menetapkan Tracking URI ke localhost sesuai kriteria Dicoding
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # 2. Mengaktifkan Autolog (Syarat Kriteria Basic)
    mlflow.sklearn.autolog()
    
    # 3. Set Eksperimen Lokal 
    mlflow.set_experiment("California_Housing_Basic_Local")
    
    with mlflow.start_run(run_name="Basic_RF_Model"):
        print("Memulai pelatihan model secara lokal dengan Autolog...")
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model basic berhasil dilatih dengan RMSE: {rmse:.2f}")