#!/usr/bin/env python
# coding: utf-8

import click

from datetime import date

from duration_prediction.train import train

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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