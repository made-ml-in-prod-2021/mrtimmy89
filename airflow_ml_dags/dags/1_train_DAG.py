from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from utils import default_args, DEFAULT_VOLUME


with DAG(
    '1_train',
    default_args=default_args,
    description='A DAG to download, split data, create a scaler, train a model and validate it',
    schedule_interval="@weekly",
    start_date=days_ago(2),
) as dag:

    start = DummyOperator(
        task_id='Start_the_pipeline'
    )

    check_scaler = FileSensor(
        task_id="Wait_for_scaler_creation",
        poke_interval=10,
        retries=100,
        filepath="data/models/{{ ds }}/scaler.pkl"
    )

    check_model = FileSensor(
        task_id="Wait_for_model_creation",
        poke_interval=10,
        retries=100,
        filepath="data/models/{{ ds }}/model.pkl"
    )

    check_metrics = FileSensor(
        task_id="Wait_for_metrics_creation",
        poke_interval=10,
        retries=100,
        filepath="data/models/{{ ds }}/scores.json"
    )

    load = DockerOperator(
        task_id='Download_data',
        image='airflow-load',
        command='--input-path /data/raw/{{ ds }} --temp-path /data/temp/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    preprocess = DockerOperator(
        task_id='Preprocess_features',
        image='airflow-preprocess',
        command='--temp-path /data/temp/{{ ds }} --preprocessed-path /data/preprocessed/{{ ds }} '
                '--scaler-path /data/models/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    split = DockerOperator(
        task_id='Split_data',
        image='airflow-split',
        command='--preprocessed-path /data/preprocessed/{{ ds }} --splitted-path /data/splitted/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    train = DockerOperator(
        task_id='Train_model',
        image='airflow-train',
        command='--train-path /data/splitted/{{ ds }} --model-path /data/models/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    validate = DockerOperator(
        task_id='Validate_model',
        image='airflow-validate',
        command='--val-path /data/splitted/{{ ds }} --metrics-path /data/models/{{ ds }} '
                '--model-path /data/models/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    finish = DummyOperator(
        task_id='Finish_model_training'
    )

    start >> load >> preprocess >> split >> train >> validate >> [check_scaler, check_model, check_metrics] >> finish
