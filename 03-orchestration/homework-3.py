import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from prefect import flow, task
from pathlib import Path

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

@task
def download_data(url=DATA_URL, output_path="yellow_tripdata_2023-03.parquet"):
    if not Path(output_path).exists():
        df = pd.read_parquet(url)
        df.to_parquet(output_path)
    else:
        df = pd.read_parquet(output_path)
    
    print(f"✅ Número de registros carregados: {len(df):,}")
    return df

@task
def preprocess_data(df):
    df = df.copy()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    X = df[categorical]
    y = df['duration']

    print(f"✅ Tamanho do dataframe pós-processamento: {len(df):,}")
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task
def train_model(X_train, X_val, y_train, y_val):
    # Converte para dicionários
    train_dicts = X_train.to_dict(orient='records')
    val_dicts = X_val.to_dict(orient='records')

    # Vetorização com DictVectorizer
    dv = DictVectorizer()
    X_train_vect = dv.fit_transform(train_dicts)
    X_val_vect = dv.transform(val_dicts)

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train_vect, y_train)

        y_pred = model.predict(X_val_vect)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Loga no MLflow
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("intercept", model.intercept_)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Intercepto: {model.intercept_:.2f}")
        print(f"✅ RMSE: {rmse:.2f}")

    return rmse

@flow
def main_flow():
    print("🚀 Iniciando pipeline com Prefect...")
    df = download_data()
    X_train, X_val, y_train, y_val = preprocess_data(df)
    train_model(X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    # 🔗 Configura o MLflow para apontar para o servidor local
    #mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nyc-taxi-prefect")

    main_flow()
