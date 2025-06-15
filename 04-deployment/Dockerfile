FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Instala dependências compatíveis com o model.bin
RUN pip install numpy==1.21.6 pandas pyarrow scikit-learn==1.0.2

COPY starter.py .

ENTRYPOINT ["python", "starter.py"]
