"""Hyperparameter tuning and model training

This script tunes LightGBM hyperparameters with the use
of Optuna library (this part of the code is presented as a comment)
and trains the model on the processed input data using the configuration
of tuned hyperparameters.

Script must be run after data.py or using backup files
x.pt, y.pt
"""

import sklearn
import sklearn.model_selection
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import sklearn.metrics
import sklearn.datasets

import optuna
import lightgbm
import joblib
import final_project.data as lib
import os
import pandas as pd
import numpy as np
import yaml

if __name__ == "__main__":

    # read config
    config = lib.read_yaml("config.yaml")
    path_data = config["path"]
    path_models = f"{path_data}/models"
    os.makedirs(path_models, exist_ok=True)

    # load the processed data
    X = pd.read_parquet(f"{path_data}/processed/x.pt")
    y = pd.read_parquet(f"{path_data}/processed/y.pt")

    # parameters have been already tuned by Optuna algorithm (regular)
    # and saved to config
    lightgbm_params = config["model"]

    # To conduct the whole tuning process (which is fairly time-consuming)
    # One would need to unhash the code part responsible for the algorithm of interest
    # (Regular Optuna or LightGBM tuner, both implementations start with numbered title and a lot of #s)
    # And run the code with ONE of the parts unhashed

    ###################################################################################
    ####
    #### 1. Regular Optuna framework for hyperparameter tuning
    ####
    ###################################################################################

    # commented functions below are used by Optuna algorithm
    # it samples a configuration of hyperparameters
    # and calculates the cross-validated value of accuracy
    # for the corresponding model

    # def objective(trial):
    #     """ Function that is optimized to tune hyperparameters of model
    #
    #     Uses binary log-loss as metric and the value of the optimized function is F-score
    #     """
    #     data, target = X, y
    #     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, shuffle=False)
    #     dtrain = lightgbm.Dataset(train_x, label=train_y)

    #     param = {
    #         'objective': 'binary',
    #         'metric': 'binary_logloss',
    #         'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
    #         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    #         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    #         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    #         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
    #         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
    #         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    #         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    #     }

    #     n_iter = config['optimizer']['n_iter']
    #     gbm = lightgbm.train(param, dtrain)
    #     preds = gbm.predict(test_x)
    #     pred_labels = np.rint(preds)
    #     f1_score = sklearn.metrics.f1_score(test_y, pred_labels)
    #     return f1_score

    # Below the code part that conducts the tuning process is presented

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=n_iter)         # 100 iterations were conducted

    # print("Number of finished trials: {}".format(len(study.trials)))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: {}".format(trial.value))

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    # lightgbm_params = study.best_params
    # config['model'] = study.best_params

    ###################################################################################
    ####
    #### 2. Hyperparameter tuning with LightGBM Tuner
    ####
    ###################################################################################

    # The commented procedure below is the tuning of parameters
    # with LightGBM Tuner, Optuna's framework-specific algorithm, which
    # has also been considered as a tuning option

    # import optuna.integration.lightgbm as lgb

    # dtrain = lgb.Dataset(X, label=y)

    # params = {
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
    #     'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    #     'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    #     'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    #     'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
    #     'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
    #     'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    #     'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    # }

    # tuner = lgb.LightGBMTunerCV(
    #     params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=TimeSeriesSplit(n_splits=5)
    # )

    # tuner.run()

    # print("Best score:", tuner.best_score)
    # best_params = tuner.best_params
    # print("Best params:", best_params)
    # print("  Params: ")
    # for key, value in best_params.items():
    #     print("    {}: {}".format(key, value))

    # lightgbm_params = best_params
    # config['model'] = best_params

    ###################################################################################

    # after either of the approaches is used fro the tuning of hyperparameters, the updated config dict is saved as yaml
    lib.write_yaml(config, "config.yaml")
    # the model with tuned parameters is trained
    lightgbm_classifier = lightgbm.LGBMClassifier(**lightgbm_params)
    lightgbm_classifier.fit(
        X, np.ravel(y)
    )

    # the trained model is saved
    joblib.dump(lightgbm_classifier, f"{path_models}/lightgbm_model.pkl")
