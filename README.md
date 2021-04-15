# HU_MLiM_Final_Project
This repository contains the model developed by my team as a final project in the course "Machine Learning in Marketing" offered by School of Economics and Business, Humboldt University of Berlin in Winter Semester 20/21. 

The task was to build the algorithm searching for the offering of discount coupons to the shoppers that maximizes the expected revenue of a grocery store on a successive week. Final coupon offering includes the discounts on 5 different products (the total product range is 250) for each of 2,000 shoppers, choice of discounted products and discount sizes is shopper-specific. 

The given data includes several datasets containing purchasing histories of the shoppers over the previous 90 weeks, product prices and the history of coupon offering and redemption. 

The code is written and tested in Python 3.9.1.

## Pipeline

The pipeline includes the following steps:
1. EDA, derivation of implicit categorical structure of product range with P2V-MAP (Word2Vec - skip-gram with negative sampling - applied to data on shopping baskets instead of text and the following t-SNE transformation) and clustering, creation of category-speicific and product-specific features based on coupon redemption and purchase histories, final contruction of training dataset (input and label features)
1. Tuning of the LightGBM hyperparameters with various Optuna frameworks
1. Model training
1. Building of baseline models and the result benchmarking
1. Use of the model for coupon offering optimization


## Environment variables

Set the following environment variables before running the pipeline:

| Variable    | Description                                                            | Example          |
| ----------- | -----------------------------------------------------------------------| ---------------- |
| `PATH_REPO` | path of the repository/working directory to place the folder           | `$HOME/mlim`     |
| `PATH_ENV`  | path of the virtual environment                                        | `$HOME/env-mlim` |


Before running the pipeline, set the path for the folder with config.yaml as a working directory
and set the value of config['path'] in config.yaml as the path for the folder with files from this repository.

PATH_REPO (working directory) should be te path to the folder that contatins this folder.

After all the things mentioned above are fixed, the whole pipeline can be launched as the file run.sh is run.
