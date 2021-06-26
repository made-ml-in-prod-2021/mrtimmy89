from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import default_args, DEFAULT_VOLUME


with DAG(
    '0_generate_data',
    default_args=default_args,
    description='A DAG to generate synthetic data',
    schedule_interval='@weekly',
    start_date=days_ago(2),
) as dag:
    start = DummyOperator(task_id='Start_data_generation')
    download = DockerOperator(
        task_id='Generate_entries',
        image='airflow-generate',
        command='--output-path /data/raw/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME],
    )
    finish = DummyOperator(
        task_id='Finish_data_generation'
    )

    start >> download >> finish
