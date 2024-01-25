#!/usr/bin/env python
# coding: utf-8

import pickle

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

df_train = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet')
df_val = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet')

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

with open('../models/lin_reg.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)
