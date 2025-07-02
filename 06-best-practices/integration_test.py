import os
import pandas as pd
from datetime import datetime
import os

# Helper de data
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Dados de entrada
data = [
    (None, None, dt(1, 1), dt(1, 10)),        # 9 min
    (1, 1, dt(1, 2), dt(1, 10)),              # 8 min
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),     # 0.98 min
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),         # 1441 min
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

# Parâmetros S3
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}

input_file = "s3://nyc-duration/in/2023-01.parquet"
output_file = "s3://nyc-duration/out/2023-01.parquet"

# Salva os dados de entrada
df_input.to_parquet(
    input_file,
    engine="pyarrow",
    compression=None,
    index=False,
    storage_options=options
)

# Executa o batch.py via linha de comando
exit_code = os.system("python batch.py 2023 1")
assert exit_code == 0, "batch.py failed"

# Lê o arquivo de saída
df_result = pd.read_parquet(output_file, storage_options=options)

# Soma das predições
print("Predicted durations:")
print(df_result)

total_duration = df_result['predicted_duration'].sum()
print("Total predicted duration:", round(total_duration, 2))
