import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# 1. Inisialisasi DagsHub (Ganti dengan kredensialmu)
# Pastikan kamu sudah membuat repository kosong di dagshub.com
REPO_OWNER = "AnqapCode"
REPO_NAME = "Eksperimen_SML_Arif"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def load_processed_data():
    """Memuat data yang sudah diproses dari Kriteria 1."""
    train_df = pd.read_csv('california_housing_preprocessing/train_processed.csv')
    test_df = pd.read_csv('california_housing_preprocessing/test_processed.csv')
    
    X_train = train_df.drop('median_house_value', axis=1)
    y_train = train_df['median_house_value']
    X_test = test_df.drop('median_house_value', axis=1)
    y_test = test_df['median_house_value']
    
    return X_train, X_test, y_train, y_test

def create_artifacts(model, X_train, y_test, y_pred):
    """Membuat minimal 2 artefak tambahan (Advance) berupa plot gambar."""
    os.makedirs("artifacts_temp", exist_ok=True)
    
    # Artefak 1: Feature Importance Plot
    plt.figure(figsize=(10, 6))
    feature_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("artifacts_temp/feature_importance.png")
    plt.close()

    # Artefak 2: Actual vs Predicted Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Housing Prices")
    plt.tight_layout()
    plt.savefig("artifacts_temp/actual_vs_predicted.png")
    plt.close()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()

    # ------------------------------------------------------------------
    # KRITERIA BASIC/SKILLED: Menyimpan ke Localhost
    # Aktifkan baris di bawah ini dan matikan/comment dagshub.init() di atas 
    # jika ingin tracking ke komputer lokal (buka terminal baru dan ketik: mlflow ui)
    # ------------------------------------------------------------------
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Menentukan nama eksperimen di MLflow
    mlflow.set_experiment("California_Housing_Manual_Tuning")
    
    # Daftar kombinasi hyperparameter yang akan diuji
    param_grid = [
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 50, 'max_depth': 20},
        {'n_estimators': 100, 'max_depth': 20},
        {'n_estimators': 75, 'max_depth': 25}
    ]
    
    print("Memulai Hyperparameter Tuning dan Logging 5 eksperimen...")

    for i, params in enumerate(param_grid):
        # Nama run agar mudah dibaca di dashboard MLflow
        run_name = f"RF_est{params['n_estimators']}_depth{params['max_depth']}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n[{i+1}/{len(param_grid)}] Menjalankan eksperimen: {run_name}")
            
            # 2. Inisialisasi dan Training Model dengan parameter saat ini
            model = RandomForestRegressor(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # 3. Prediksi dan Evaluasi Metrik
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # 4. Manual Logging MLflow (Parameter, Metrik, Model)
            mlflow.log_params(params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            
            # Membuat signature agar rapi
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)
            
            # 5. Membuat dan Melogging Artefak Tambahan
            create_artifacts(model, X_train, y_test, y_pred)
            mlflow.log_artifacts("artifacts_temp", artifact_path="evaluation_plots")