#!/usr/bin/env python
# coding: utf-8

import pickle
from datetime import date

import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline



def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60 # turn seconds to minutes

    df = df[(df.duration >= 1) & (df.duration <= 60)] # remove very short and very long rides

    categorical = ['PULocationID', 'DOLocationID'] # pick up location and drop off location
    df[categorical] = df[categorical].astype(str) # turn into strings
    
    return df

def train(train_month: date, validation_month: date, model_output_path: str) -> None:
    url_template = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = url_template.format(year=train_month.year, month=train_month.month)
    val_url = url_template.format(year=validation_month.year, month=validation_month.month)

    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    pipeline = make_pipeline(
        DictVectorizer(),
        LinearRegression()
    )

    # Will use only 3 features fro the model
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

    print(f'rmse = {rmse}')

    with open(model_output_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

@click.command()
@click.option('--train-month', required=True, help='Training month in YYYY-MM format')
@click.option('--validation-month', required=True, help='Validation month in YYYY-MM format')
@click.option('--model-output-path', required=True, help='Path where the trained model will be saved')
def run(train_month, validation_month, model_output_path):
    train_year, train_month = train_month.split('-')
    train_year = int(train_year)
    train_month = int(train_month)

    val_year, val_month = validation_month.split('-')
    val_year = int(val_year)
    val_month = int(val_month)

    train_month = date(year=train_year, month=train_month, day=1)
    validation_month = date(year=val_year, month=val_month, day=1)

    train(
        train_month=train_month,
        validation_month=validation_month,
        model_output_path=model_output_path
    )

if __name__ == '__main__':
    run()