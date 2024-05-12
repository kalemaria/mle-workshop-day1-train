#!/usr/bin/env python
# coding: utf-8

import pickle
from datetime import date

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import logging
logger = logging.getLogger(__name__)

def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a Parquet file into a DataFrame and preprocess the data.

    This function reads a Parquet file, computes the duration of each taxi trip,
    filters the trips based on duration criteria, and converts certain categorical
    columns to string type.

    Parameters:
    filename (str): The path to the Parquet file.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    logger.info(f"Reading data from {filename}")

    try:
        df = pd.read_parquet(filename)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60 # turn seconds to minutes

        df = df[(df.duration >= 1) & (df.duration <= 60)] # remove very short and very long rides

        categorical = ['PULocationID', 'DOLocationID'] # pick up location and drop off location
        df[categorical] = df[categorical].astype(str) # turn into strings
        
        return df
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        raise

def train(train_month: date, validation_month: date, model_output_path: str) -> None:
    """
    Train a linear regression model for predicting taxi trip durations.

    This function trains a model using data from specified months, evaluates it,
    and saves the trained model to a file. It reads data, preprocesses it,
    fits a linear regression model within a pipeline, and evaluates the model
    using RMSE. The trained model is then saved to the specified path.

    Parameters:
    train_month (date): The month for training data.
    val_month (date): The month for validation data.
    model_output_path (str): The file path to save the trained model.

    Returns:
    None
    """
    logger.info(f"Training model for {train_month} and validating for {validation_month}")

    try:
        url_template = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        train_url = url_template.format(year=train_month.year, month=train_month.month)
        val_url = url_template.format(year=validation_month.year, month=validation_month.month)

        logger.debug(f"URL for training data: {train_url}")
        logger.debug(f"URL for validation data: {val_url}")

        df_train = read_dataframe(train_url)
        df_val = read_dataframe(val_url)

        pipeline = make_pipeline(
            DictVectorizer(),
            LinearRegression()
        )

        # Will use only 3 features for the model
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']

        train_dicts = df_train[categorical + numerical].to_dict(orient='records')
        val_dicts = df_val[categorical + numerical].to_dict(orient='records')

        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values

        pipeline.fit(train_dicts, y_train)

        y_pred = pipeline.predict(val_dicts)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        logger.info(f'RMSE: {rmse}')
        logger.info(f"Model trained successfully. Saving model to {model_output_path}")

        with open(model_output_path, 'wb') as f_out:
            pickle.dump(pipeline, f_out)

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise