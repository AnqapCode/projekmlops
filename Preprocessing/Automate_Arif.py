import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(url):
    """Memuat dataset dari URL."""
    print("Memuat data raw...")
    return pd.read_csv(url)

def clean_data(df):
    """Menangani nilai kosong dan mengubah data kategorikal."""
    print("Menangani missing values dan melakukan label encoding...")
    # Mengisi nilai kosong dengan median
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    # Label encoding untuk kolom ocean_proximity
    label_encoder = LabelEncoder()
    df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])
    return df

def split_and_scale(df):
    """Memisahkan fitur/target, split train/test, dan standarisasi."""
    print("Memisahkan dan melakukan standarisasi data...")
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standarisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Menggabungkan kembali fitur yang sudah discaling dengan target
    train_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_processed['median_house_value'] = y_train.values
    
    test_processed = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_processed['median_house_value'] = y_test.values
    
    return train_processed, test_processed

def save_data(df_raw, train_processed, test_processed):
    """Menyimpan data mentah dan data yang sudah diproses ke dalam folder yang sesuai."""
    print("Menyimpan data...")
    # Sesuai kriteria folder submission
    os.makedirs('../california_housing_raw', exist_ok=True)
    os.makedirs('california_housing_preprocessing', exist_ok=True)
    
    # Simpan raw data
    df_raw.to_csv('../california_housing_raw/housing_raw.csv', index=False)
    
    # Simpan data yang sudah diproses
    train_processed.to_csv('california_housing_preprocessing/train_processed.csv', index=False)
    test_processed.to_csv('california_housing_preprocessing/test_processed.csv', index=False)

if __name__ == "__main__":
    # URL Dataset California
    DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    
    # Menjalankan fungsi-fungsi secara berurutan
    raw_df = load_data(DATA_URL)
    df_clean = clean_data(raw_df.copy())
    train_data, test_data = split_and_scale(df_clean)
    
    save_data(raw_df, train_data, test_data)