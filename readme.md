
# Wine Quality

![wine](wine.png)

This is the last project for the course 'Fundamentos da Inteligência Artificial". The main purpose of this project is to implement a MultiLayerPerceptron (MLP) model using data from UCI Machine Learning Repository.

For this project I chose the Wine Quality dataset that describe more than 4898 different variants of 'Vinho verde' wine. In this project I will train a MLP model that can predict the quality of a given wine.

> dataset link : https://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Project Structure

```

├── readme.md          <- The top-level README.
│
├── data
│   ├── final          <- Model-ready dataset.
│   ├── intermediate   <- Intermediate data for EDA and data preparation.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained models.
│
├── notebooks          <- Jupyter notebooks
│
├── reports            <- Reports created.
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── make_features.py
│   │
│   └── models         <- Scripts to train models and then use trained models to make predictions
│       └── train_model.py
│
└── environment.yml    <- The yml file for setting up the conda environment with all requirements.

```