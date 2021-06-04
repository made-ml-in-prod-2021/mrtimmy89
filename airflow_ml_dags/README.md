# AirFlow for ML<br><br>
3 DAGs representing:<br>
- 0: data generation pipeline
- 1: training on given data by means of data transformation (via StandardScaler), splitting into train and validation datasets, training a model (LogisticRegression) 
and then evaluating the model with some metrics
- 2: prediction pipeline (for its correct work the directory "input" containing the input.csv file should be created in the "data" directory)
<br<<br>
# Correct run<br><br>
docker-compose up --build
<br><br>
# Correct stop<br><br>
docker-compose down
