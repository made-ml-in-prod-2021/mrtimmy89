from datetime import timedelta

DEFAULT_VOLUME = 'C:/Users/Artem/Desktop/MADE/MLOPS/airflow_ml_dags/data:/data'

default_args = {
    'owner': 'Artem Akopian',
    'depends_on_past': False,
    'email': ['akopian.artyom@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}
