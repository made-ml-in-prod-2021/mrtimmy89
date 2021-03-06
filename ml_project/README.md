## Data:

Dataset ["heart-disease-uci"](https://www.kaggle.com/ronitf/heart-disease-uci) is loaded from a kaggle competition.
<br>
<br>
## Preresquistes:

Python 3.7
<br>
virtualenv (pip install virtualenv)
<br>
<br>
## Installation:

pip install -r requirements.txt
<br>
<br>
## Usage:

train: python train_pipeline.py configs/train_config.yaml
<br>
<br>
predict: python predict_pipeline.py config/predict_config.yaml
<br>
<br>
## PEP 8 testing:

pylint --output-format=colorized -v src
<br>
<br>
## Roadmap:

|  |Задание|Разбалловка|
|---|-------------------------------------------------------------------------------------------------------------|:-------------:|
|1.|Создана ветка homework1|+1|
|2.|В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе|+2|
|3.|EDA выполнен и закоммичен|+2|
|4.|Проект имеет модульную структуру|+2|
|5.|Использованы логгеры|+2|
|6.|Написаны тесты на отдельные модули и на прогон всего пайплайна|+3|
|7.|Для тестов генерируются синтетические данные|+3|
|8.|В наличии 2 корректные конфигурации .yaml конфигов|+3|
|9.|Используются датаклассы для сущностей из конфига|+3|
|10.|Кастомный трансформер не сделан|+0|
|11.|В readme записано, как учить модель|+3|
|12.|В readme записано, как получить предикт модели|+3|
|13.|Использование hydra|+1<br>(неполное решение)|
|14.|CI не настроен|+0|
|15.|Самооценка / самокопание / самоанализ|+1|
<br>
<br>
По предложенной разбалловке получилось 29 баллов. Надеюсь, нигде не ошибся.
<br>
<br>

## Project structure:

```
├── README.md             <- The top-level README for developers using this project.
│
├── requirements.txt      <- The requirements file for reproducing the analysis environment.
│
├── setup.py              <- File for install
│
├─ data_generator.py  	  <- File for creating a synthetic dataset for test│
│
├── trani_pipeline.py     <- File for train_pipeline
│
├── predict_pipeline.py   <- File for predict pipeline
│
├── setup.py              <- File for install
│
├── configs               <- Configuration files.
│
├── docs                  <- A default Sphinx project
│
├── data
│   │
│   └── raw               <- The original, immutable data dump.
│
├── models                <- Trained and serialized models.
│
├── notebooks             <- Jupyter notebooks.
│
├── predictions           <- Outputs from predict pipeline.
│
├── reports               <- Report from training model.
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
