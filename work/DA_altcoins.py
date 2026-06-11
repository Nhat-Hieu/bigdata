#!/usr/bin/env python
# coding: utf-8

# =========================================
# IMPORT LIBRARIES
# =========================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

import boto3
from botocore.client import Config

# =========================================
# INIT SPARK SESSION (LIGHTWEIGHT)
# =========================================

spark = SparkSession.builder \
    .appName("SV3_Altcoins_ETL_Optimized") \
    .getOrCreate()

print("Spark Started Successfully")

# =========================================
# MINIO CONFIG
# =========================================

hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "admin")
hadoop_conf.set("fs.s3a.secret.key", "password123")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

hadoop_conf.set(
    "fs.s3a.aws.credentials.provider",
    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
)

print("MinIO Config Completed")

# =========================================
# FIXED SCHEMA (IMPORTANT OPTIMIZATION)
# =========================================

schema = StructType([
    StructField("symbol", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
])

# =========================================
# LOAD DATA (NO inferSchema)
# =========================================

df = spark.read.csv(
    "s3a://crypto-raw-data/altcoins_500d.csv",
    header=True,
    schema=schema
)

print("Raw Data Loaded")

df = df.cache()
df.count()   # materialize cache

# =========================================
# TIMESTAMP CLEANING
# =========================================

df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

df = df.orderBy("symbol", "timestamp")

print("Timestamp standardized")

# =========================================
# WINDOW (SINGLE BASE WINDOW - OPTIMIZED)
# =========================================

w = Window.partitionBy("symbol").orderBy("timestamp")

w10 = w.rowsBetween(-9, 0)
w14 = w.rowsBetween(-13, 0)
w60 = w.rowsBetween(-59, 0)

# =========================================
# GAP DETECTION
# =========================================

df = df.withColumn("prev_time", F.lag("timestamp").over(w))

df = df.withColumn(
    "diff_day",
    F.datediff(F.col("timestamp"), F.col("prev_time"))
)

gap_count = df.filter(F.col("diff_day") > 1).count()
print("Gap Count:", gap_count)

# =========================================
# FEATURE ENGINEERING (OPTIMIZED PASS)
# =========================================

# price lag
df = df.withColumn("close_lag10", F.lag("close", 10).over(w))

# MA features
df = df \
    .withColumn("MA10", F.avg("close").over(w10)) \
    .withColumn("MA60", F.avg("close").over(w60))

# ROC + MOM
df = df.withColumn(
    "ROC",
    (F.col("close") - F.col("close_lag10")) /
    F.col("close_lag10") * 100
)

df = df.withColumn(
    "MOM",
    F.col("close") - F.col("close_lag10")
)

# =========================================
# RSI (OPTIMIZED)
# =========================================

df = df.withColumn(
    "change",
    F.col("close") - F.lag("close").over(w)
)

df = df.withColumn(
    "gain",
    F.when(F.col("change") > 0, F.col("change")).otherwise(0)
)

df = df.withColumn(
    "loss",
    F.when(F.col("change") < 0, -F.col("change")).otherwise(0)
)

df = df \
    .withColumn("avg_gain", F.avg("gain").over(w14)) \
    .withColumn("avg_loss", F.avg("loss").over(w14))

df = df.withColumn(
    "RS",
    F.when(F.col("avg_loss") == 0, None)
     .otherwise(F.col("avg_gain") / F.col("avg_loss"))
)

df = df.withColumn(
    "RSI",
    F.when(F.col("avg_loss") == 0, 100)
     .when(F.col("avg_gain") == 0, 0)
     .otherwise(100 - (100 / (1 + F.col("RS"))))
)

print("RSI Done")

# =========================================
# STOCHASTIC (OPTIMIZED)
# =========================================

df = df \
    .withColumn("highest_high", F.max("high").over(w14)) \
    .withColumn("lowest_low", F.min("low").over(w14))

df = df.withColumn(
    "stoch_k",
    F.when(
        (F.col("highest_high") - F.col("lowest_low")) == 0,
        None
    ).otherwise(
        (F.col("close") - F.col("lowest_low")) /
        (F.col("highest_high") - F.col("lowest_low")) * 100
    )
)

df = df.withColumn(
    "stoch_d",
    F.avg("stoch_k").over(w.rowsBetween(-2, 0))
)

print("Stochastic Done")

# =========================================
# LABEL
# =========================================

df = df.withColumn(
    "label",
    F.when(F.col("MA10") > F.col("MA60"), 1).otherwise(0)
)

df.groupBy("symbol", "label").count().show()

# =========================================
# FINAL TABLE
# =========================================

final_df = df.select(
    "symbol",
    "timestamp",
    "open", "high", "low", "close", "volume",
    "MA10", "MA60",
    "ROC", "MOM",
    "RSI",
    "stoch_k", "stoch_d",
    "label"
).dropna()

print("Before DropNA:", df.count())
print("After DropNA:", final_df.count())

# =========================================
# MINIO CLIENT
# =========================================

s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="admin",
    aws_secret_access_key="password123",
    config=Config(signature_version="s3v4")
)

bucket_name = "crypto-feature-table"

if bucket_name not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
    s3.create_bucket(Bucket=bucket_name)
    print("Bucket Created")
else:
    print("Bucket Exists")

# =========================================
# SAVE OUTPUT
# =========================================

final_df.write \
    .mode("overwrite") \
    .parquet("s3a://crypto-feature-table/altcoin_features/")

print("Saved Successfully")

# =========================================
# VERIFY OUTPUT (LIGHTWEIGHT)
# =========================================

verify_df = spark.read.parquet(
    "s3a://crypto-feature-table/altcoin_features/"
)

print("Rows Written:", verify_df.count())
verify_df.show(20, False)