from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="crypto_pipeline",
    start_date=datetime(2026, 6, 1),
    schedule="@daily",
    catchup=False
) as dag:

    run_spark = BashOperator(
        task_id="run_spark",
        bash_command="""
docker exec -e PYSPARK_SUBMIT_ARGS="" pyspark-jupyter spark-submit \
--packages org.apache.hadoop:hadoop-aws:3.4.2 \
/home/jovyan/work/DA.py
""")