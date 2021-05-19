## Project structure:

```
├── README.md             <- The top-level README for developers using this project.
│
├── requirements.txt      <- The requirements file for reproducing the analysis environment.
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
