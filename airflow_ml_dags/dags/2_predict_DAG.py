from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from utils import default_args, DEFAULT_VOLUME


with DAG(
    '2_prediction',
    default_args=default_args,
    description='A DAG to make a prediction',
    schedule_interval="@daily",
    start_date=days_ago(2),
) as dag:

    start = DummyOperator(
        task_id='Start_predicting'
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

    check_prediction = FileSensor(
        task_id="Wait_for_prediction_creation",
        poke_interval=10,
        retries=100,
        filepath="data/predictions/{{ ds }}/predictions.csv"
    )

    predict = DockerOperator(
        task_id='Make_prediction',
        image='airflow-predict',
        command='--input-path /data/input/{{ ds }} --prediction-path /data/predictions/{{ ds }} '
                '--scaler-path /data/models/{{ ds }} --model-path /data/models/{{ ds }}',
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    finish = DummyOperator(
        task_id='Finish_predicting'
    )

    start >> [check_scaler, check_model] >> predict >> check_prediction >> finish
