## The docker image optimization:
<br>
The optimization has been done with python:3.8-slim  which caused the reduction of the Image size from 1,4 Gb to 628,57 Mb
<br>
<br>

## Running from docker hub:
<br>
docker pull mrtimmy89/mlops2
<br>
docker run -p 80:80 mrtimmy89/mlops2 (for example, without giving a name)
<br>
<br>

## Running from the CLI:
<br>
uvicorn app:app --reload
<br>
<br>

## Requests are made the following way:
<br>
python make_request.py
<br>
<br>

## Covering with tests and pylint:
<br>
pylint
<br>
pytest -v
<br>
<br>

## Roadmap:

|  |Задание|Разбалловка|
|---|-------------------------------------------------------------------------------------------------------------|:-------------:|
|0.|Созданы ветка homework2 и папка online_inference|-|
|1.|inference модели "обернут" в rest сервис с использованием FastAPI|+3|
|2.|Написан тест для /predict|+3|
|3.|Написан скрипт, который может делать запросы к сервису |+2|
|4.|Валидация не делалась|+0|
|5.|Написан dockerfile, на его основе собран образ|+4|
|6.|Размер docker image оптимизирован|+3|
|7.|Образ опубликован в https://hub.docker.com/|+2|
|8.|В readme написаны корректные команды docker pull/run|+1|
|9.|Проведена самооценка|+1|
<br>
<br>
По предложенной разбалловке получилось 19 баллов.
<br>
<br>

## Project structure:

```
├── README.md             <- The top-level README for developers using this project.
│
├── requirements.txt      <- The requirements file for reproducing the analysis environment.
│
├── Dockerfile            <- The Docker Image.
│
├── setup.py              <- File for installation.
│
├── app.py 	              <- FastAPI application
│
├── make_request.py       <- Script for generating requests to the application
│
├── predict_pipeline.py   <- File for predict pipeline
│
├── configs               <- Configuration files.
│
├── data
│   │
│   └── raw               <- The original, immutable data dump.
│   │
│   └── processed         <- The synthetic dataset (the original one without the "target" column)
│
├── models                <- Trained and serialized models.
│
├── predictions           <- Outputs from application.
│
├── src                   <- Source code for use in this project.
│   │
│   ├── __init__.py       <- Makes src a Python module
│   │
│   ├── entities          <- configuration dataclasses for type checking
│   │
│   ├── data              <- code to generate and transform data
│   │
│   ├── features          <- code to work with features
│   │
│   └── models            <- code to work with models
|
└── tests                 <- unit tests
```
