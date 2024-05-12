# Taxi Trip Duration Prediction - Training Pipeline

This project focuses on converting a Jupyter notebook into a training pipeline to predict taxi trip durations. It utilizes the NYC TLC Trip Record Data to model and forecast travel times in New York City.

This project was done during the Machine Learning Engineering Course (Day 1) at Constructor Academy's Data Science Bootcamp.
This is the original repository of the workshop: https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop/

## Installation instructions

Pre-requisites:

- Docker
- Python 3.10
- Pipenv


```bash
# note that the address is fake
git clone git@github.com:kalemaria/mle-workshop-day1-train.git
pipenv install --dev
```

## Project Structure

The project is organized into several directories and files, each serving a specific purpose:

- `duration_prediction/`: the source code for the training pipeline and other Python scripts
- `tests/`: test cases and test data
- `notebooks/`: notebooks for experimentation and one-off analyses
- `models/`: Where trained model files are saved
- `README.md`: The main documentation file providing an overview and instructions for the project
- `Makefile`: A script for automating common tasks like testing, running, or deploying the project


## Usage

Running it:

```bash
TRAIN="2022-01"
VAL="2022-02"

pipenv run python \
    -m duration_prediction.main \
    --train-month="${TRAIN}" \
    --validation-month="${VAL}" \
    --model-output-path="./models/model-${TRAIN}.bin"
```

Run only one test:

```bash
pipenv run python -m unittest tests.test_train
```

Running all tests in `tests/`

```bash
pipenv run python -m unittest discover -s tests
```

Or

```bash
make tests
```