from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import io
import boto3
from botocore.client import Config

def crawl_realtime_job():
    print("Bắt đầu lấy dữ liệu phút mới nhất...")
    exchange = ccxt.bitstamp()

    btc_ohlcv = exchange.fetch_ohlcv('BTC/USD', '1m', limit=1000)
    df_btc = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # --- 1. BỘ LỌC RÁC ---
    # Ép các cột giá về chuẩn số, loại bỏ dòng chứa rác (như chữ "60s")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df_btc[col] = pd.to_numeric(df_btc[col], errors='coerce')
    df_btc = df_btc.dropna(subset=numeric_cols)
    
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    df_btc['timestamp'] = df_btc['timestamp'] + pd.Timedelta(hours=7)
    
    s3_client = boto3.client(
        's3',
        endpoint_url="http://minio:9000",
        aws_access_key_id="admin",
        aws_secret_access_key="password123",
        config=Config(signature_version='s3v4')
    )
    
    # --- 2. ĐOẠN CODE TẠO BUCKET BẠN VIẾT TRÊN JUPYTER ---
    BUCKET_NAME = "crypto-raw-data"
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' đã tồn tại.")
    except:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        print(f"Đã tự động tạo mới Bucket: '{BUCKET_NAME}' trên MinIO.")

    # Ghi file
    csv_buffer = io.StringIO()
    df_btc.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key='bitcoin_1m.csv', Body=csv_buffer.getvalue())
    print(f"Hoàn thành cập nhật {len(df_btc)} dòng dữ liệu realtime sạch!")

default_args = {
    'owner': 'minhquy',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1), 
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

with DAG(
    'crypto_1m_realtime_pipeline',
    default_args=default_args,
    schedule_interval='* * * * *', 
    catchup=False,
    tags=['crypto', 'realtime'],
) as dag:

    task_crawl = PythonOperator(
        task_id='crawl_and_push_minio',
        python_callable=crawl_realtime_job
    )