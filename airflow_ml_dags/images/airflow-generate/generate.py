import os
import click
from random import randint

from sklearn.datasets import load_breast_cancer


@click.command("generate")
@click.option("--output-path")
def generate(output_path: str):
    """
    We generate n entries which should be less than known dataset size 569
    :param output_dir:
    :return:
    """
    n_entries = randint(10, 568)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    os.makedirs(output_path, exist_ok=True)
    X[:n_entries].to_csv(os.path.join(output_path, "data.csv"))
    y[:n_entries].to_csv(os.path.join(output_path, "target.csv"))

if __name__ == '__main__':
    generate()
