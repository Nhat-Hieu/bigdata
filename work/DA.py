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
# INIT SPARK SESSION (OPTIMIZED)
# =========================================

spark = SparkSession.builder \
    .appName("SV3_Crypto_ETL") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

print("Spark Started Successfully")


# =========================================
# MINIO S3A CONFIG
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

print("MinIO Configuration Completed")


# =========================================
# LOAD RAW DATA FROM MINIO (FIXED)
# =========================================

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
])

df = spark.read.csv(
    "s3a://crypto-raw-data/bitcoin_1m.csv",
    header=True,
    schema=schema
)

df = df.cache()

print("Raw Dataset Loaded")
print("Total Rows:", df.count())

df.show(20)
df.printSchema()


# =========================================
# NULL CHECK
# =========================================

print("NULL CHECK")

df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()


# =========================================
# DUPLICATE CHECK
# =========================================

total_rows = df.count()

unique_rows = df.dropDuplicates(["timestamp"]).count()

print("Total Rows:", total_rows)
print("Unique Rows:", unique_rows)
print("Duplicates:", total_rows - unique_rows)


# =========================================
# STANDARDIZE TIMESTAMP
# =========================================

df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
df = df.orderBy("timestamp")

df.select(
    F.min("timestamp").alias("min_time"),
    F.max("timestamp").alias("max_time")
).show(truncate=False)


# =========================================
# GAP DETECTION
# =========================================

w = Window.orderBy("timestamp")

df = df.withColumn("prev_time", F.lag("timestamp").over(w))

df = df.withColumn(
    "diff_min",
    (F.unix_timestamp("timestamp") - F.unix_timestamp("prev_time")) / 60
)

df.select("timestamp", "prev_time", "diff_min").show(20, False)

gap_count = df.filter(F.col("diff_min") > 1).count()

print("Gap Count:", gap_count)


# =========================================
# MA10 & MA60
# =========================================

w10 = Window.orderBy("timestamp").rowsBetween(-9, 0)
w60 = Window.orderBy("timestamp").rowsBetween(-59, 0)

df = df.withColumn("MA10", F.avg("close").over(w10))
df = df.withColumn("MA60", F.avg("close").over(w60))

df.select("timestamp", "close", "MA10", "MA60").show(20, False)


# =========================================
# ROC + MOMENTUM
# =========================================

df = df.withColumn("close_lag10", F.lag("close", 10).over(w))

df = df.withColumn(
    "ROC",
    (F.col("close") - F.col("close_lag10")) / F.col("close_lag10") * 100
)

df = df.withColumn(
    "MOM",
    F.col("close") - F.col("close_lag10")
)

df.select("close", "close_lag10", "ROC", "MOM").show(20, False)


# =========================================
# RSI 14
# =========================================

w14 = Window.orderBy("timestamp").rowsBetween(-13, 0)

df = df.withColumn("change", F.col("close") - F.lag("close").over(w))

df = df.withColumn(
    "gain",
    F.when(F.col("change") > 0, F.col("change")).otherwise(0)
)

df = df.withColumn(
    "loss",
    F.when(F.col("change") < 0, -F.col("change")).otherwise(0)
)

df = df.withColumn("avg_gain", F.avg("gain").over(w14))
df = df.withColumn("avg_loss", F.avg("loss").over(w14))

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

print("RSI Created")

df.select(
    "timestamp", "close", "RSI"
).show(20, False)


# =========================================
# STOCHASTIC OSCILLATOR
# =========================================

df = df.withColumn("highest_high", F.max("high").over(w14))
df = df.withColumn("lowest_low", F.min("low").over(w14))

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

w3 = Window.orderBy("timestamp").rowsBetween(-2, 0)

df = df.withColumn("stoch_d", F.avg("stoch_k").over(w3))

print("Stochastic Created")

df.select(
    "timestamp", "stoch_k", "stoch_d"
).show(20, False)


# =========================================
# BUY / SELL LABEL
# =========================================

df = df.withColumn(
    "label",
    F.when(F.col("MA10") > F.col("MA60"), 1).otherwise(0)
)

print("Buy/Sell Label Created")

df.groupBy("label").count().show()


# =========================================
# FEATURE TABLE
# =========================================

final_df = df.select(
    "timestamp",
    "open", "high", "low", "close", "volume",
    "MA10", "MA60",
    "ROC", "MOM",
    "RSI",
    "stoch_k", "stoch_d",
    "label"
)

print("Rows Before DropNA:", final_df.count())

final_df = final_df.dropna()

print("Rows After DropNA:", final_df.count())


# =========================================
# CREATE MINIO BUCKET
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
    print("Bucket Already Exists")


# =========================================
# SAVE FEATURE TABLE
# =========================================

final_df.write \
    .mode("overwrite") \
    .parquet("s3a://crypto-feature-table/features/")

print("Feature Table Saved")


# =========================================
# VERIFY OUTPUT
# =========================================

verify_df = spark.read.parquet(
    "s3a://crypto-feature-table/features/"
)

print("Rows Written:", verify_df.count())

verify_df.show(30, False)
verify_df.printSchema()