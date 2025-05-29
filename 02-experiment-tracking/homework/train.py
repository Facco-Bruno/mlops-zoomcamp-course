import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-trip-duration")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():

        mlflow.set_tag("developer", "Bruno Facco")
        mlflow.log_param("train_data_size", X_train.shape[0])
        mlflow.log_param("val_data_size", X_val.shape[0])
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 0)
    
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        mlflow.log_param("min_samples_split", rf.get_params()['min_samples_split'])

        rf.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            rf,
            artifact_path="models_mlflow",
            registered_model_name="random-forest-regressor-simple-nys-taxi"
        )

        y_pred = rf.predict(X_val)
        
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':
    run_train()
