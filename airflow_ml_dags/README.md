# AirFlow for ML<br><br>
## Description<br>
3 DAGs representing:<br>
- 0: data generation pipeline 
- 1: training on given data by means of data transformation (via StandardScaler), splitting into train and validation datasets, training a model (LogisticRegression) 
and then evaluating the model with some metrics
- 2: prediction pipeline (for its successful work the directory "input" containing the input.csv file should be created in the "data" directory)<br><br>
All the DAGs are stored in the form of Docker images.
<br><br>
## Correct run<br>
docker-compose up --build
<br><br>
## Correct stop<br>
docker-compose down
<br><br>
## Roadmap:

|  |Задание|Разбалловка|
|---|-------------------------------------------------------------------------------------------------------------|:-------------:|
|0.|Поднять локально airflow|-|
|1.|Реализовать DAG, генерирующий и сохраняющий данные|+5|
|2.|Реализовать DAG, еженедельно тренирующий модель|+10|
|3.|Реализовать DAG, ежедневно использующий модель для предсказаний|+5|
|4.|Реализовать сенсоры|+3|
|5.|Все DAG реализованы только с помощью DockerOperator|+10|
|6.|Тесты для дагов написаны|+3|
|7.|mlflow не настроен|+0|
|8.|alert не настроен|+0|
|9.|Проведена самооценка|+1|
<br>
<br>
По предложенной разбалловке получилось 37 баллов.
<br>
<br>
