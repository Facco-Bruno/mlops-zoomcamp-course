#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd

# Argumentos via linha de comando
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()

year = args.year
month = args.month

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(input_file)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Q1: Desvio padrão das predições (pode descomentar se quiser usar)
#print("Desvio padrão das predições:", y_pred.std())

# ride_id
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

output_file = f'result_{year}-{month:02d}.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Q2: Checar o tamanho do arquivo (opcional)
# import os
# file_size = os.path.getsize(output_file)
# file_size_mb = file_size / (1024 * 1024)
# print(f'Tamanho do arquivo: {file_size_mb:.2f} MB')

# Q5: imprimir média da predição
print(f'Mean predicted duration: {y_pred.mean():.2f}')
