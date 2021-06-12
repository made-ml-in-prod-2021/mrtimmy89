import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_generate_pipeline(dag_bag):
    structure = {
        "start": ["download"],
        "download": ["finish"],
        "finish": []
    }
    dags_ = dag_bag.dags["0_generate_DAG"]
    for dag, task in dags_.task_dict.items():
        assert set(structure[dag]) == task.downstream_task_ids

        
def test_train_pipeline(dag_bag):
    structure = {
        "start": ["load"],
        "load": ["preprocess"],
        "preprocess": ["split"],
        "split": ["train"],
        "train": ["validate"],
        "validate": ["check_scaler", "check_model", "check_metrics"],
        "check_scaler": ["finish"],
        "check_model": ["finish"],
        "check_metrics": ["finishl"]
    }
    dags_ = dag_bag.dags["1_train_DAG.py"]
    for dag, task in dags_.task_dict.items():
        assert set(structure[dag]) == task.downstream_task_ids
