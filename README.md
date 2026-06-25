# Crypto Data Pipeline using PySpark, Airflow and MinIO

## Project Overview

This project builds an end-to-end cryptocurrency data pipeline for collecting, processing and analyzing market data.

The system automatically collects cryptocurrency data, stores raw datasets in MinIO, processes data using PySpark, generates technical indicators and feature tables, and prepares datasets for machine learning models.

---

## Technologies

* Python
* PySpark
* Apache Airflow
* MinIO
* PostgreSQL
* Docker Compose
* Jupyter Notebook

---

## System Architecture

```text
Bitstamp API
     ↓
MinIO (Raw Data Storage)
     ↓
PySpark ETL Pipeline
     ↓
Feature Engineering
     ↓
Feature Tables (.parquet)
     ↓
Machine Learning Models
```

---

## Feature Engineering

Technical indicators generated:

* MA10
* MA60
* ROC
* MOM
* RSI
* Stochastic K
* Stochastic D

Target label:

* Buy (1)
* Sell (0)

---

## Dataset

### Bitcoin Feature Table

Columns:

* timestamp
* open
* high
* low
* close
* volume
* MA10
* MA60
* ROC
* MOM
* RSI
* stoch_k
* stoch_d
* label

### Altcoin Feature Table

Contains feature tables for multiple cryptocurrencies including:

* ETH
* XRP
* ADA
* SOL
* DOGE
* LINK
* LTC
* DOT
* AVAX
* AAVE
* BCH
* ALGO
* UNI
* SHIB

---

## Project Structure

```text
bigdata/
│
├── dags/
│   ├── crypto_pipeline.py
│   └── crypto_altcoin_pipeline.py
│
├── work/
│   ├── DA2_Read_Bitcoin.ipynb
│   ├── DA2_Process_Bitcoin.ipynb
│   ├── DA2_FE_Bitcoin.ipynb
│   ├── DA2_Label_Save_Bitcoin.ipynb
│   ├── DA2_Read_Altcoins.ipynb
│   ├── DA2_Process_Altcoins.ipynb
│   ├── DA2_FE_Altcoins.ipynb
│   └── DA2_Label_Save_Altcoins.ipynb
│
├── compose.yaml
├── README.md
└── dataset_description.txt
```

---

## Running the Project

Clone repository:

```bash
git clone https://github.com/Nhat-Hieu/bigdata.git
cd bigdata
```

Start all services:

```bash
docker compose up -d
```

---

## Access Services

### Jupyter Notebook

```text
http://localhost:8888
```

### Apache Airflow

```text
http://localhost:8080
```

### MinIO

```text
http://localhost:9001
```

Username:

```text
admin
```

Password:

```text
password123
```

---

## Output

Generated feature tables:

```text
bitcoin_features.parquet
altcoin_features.parquet
```

These datasets are used for machine learning tasks such as Buy/Sell signal prediction.

---

## Author

Hồ Tăng Nhật Hiếu

Major: Data Science and Artificial Intelligence
