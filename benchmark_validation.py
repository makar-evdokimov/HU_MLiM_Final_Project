"""Building of baseline models, cross-validation and result benchmarking

This script uses time-based split of dataset, trains our model and
some alternative models (baselines) on the training set with
the further measurement of performance on the validation set

The cross-validated scores of performance metrics
are written in yaml file (scores.yaml)

Script should be run after data.py
"""

import os
import yaml
import lightgbm
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.metrics
import final_project.data as lib
import numpy as np
import pandas as pd
import random

if __name__ == "__main__":
    # read config
    config = lib.read_yaml("config.yaml")
    path_data = config["path"]
    path_results = f"{path_data}/results"
    os.makedirs(path_results, exist_ok=True)

    # load the processed data, lists for model-specific indexing
    # and saved hyperparameters of the model

    train_set = pd.read_parquet(f"{path_data}/processed/train_set.pt")
    y = pd.DataFrame(train_set['y'])
    lightgbm_params = config["model"]

    product_features = ['max_price', 'discount', 'time_since_last_purchase_of_product', 'avg_purch_freq_shift',
                        'redemption_rate_shift']
    all_features = ['max_price', 'discount', 'disc_other_in_cat', 'time_since_last_purchase_of_product',
                    'avg_purch_freq_shift', 'redemption_rate_shift', 'disc_subst', 'disc_compl']

    # time-based train-test split for the product-level dataset and the category-level dataset

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        train_set,
        y,
        test_size=0.2, shuffle=False
    )

    logit_classifier = LogisticRegression(max_iter=100_000)
    lightgbm_classifier = lightgbm.LGBMClassifier(**lightgbm_params)

    # random guess of product purchase probabilities
    y_rnd = np.random.uniform(0, 1, y_test.shape[0])
    bs_rnd = sklearn.metrics.brier_score_loss(y_test, y_rnd)
    auc_rnd = sklearn.metrics.roc_auc_score(y_test, y_rnd)

    # logit model that is trained on the input containing only product-specific features
    # no information about categorical structure is used
    logit_classifier_prod = logit_classifier.fit(
        X_train[product_features], np.ravel(y_train)
    )
    y_pred_logit_prod = logit_classifier_prod.predict_proba(X_test[product_features])[:, 1]
    bs_logit_prod = sklearn.metrics.brier_score_loss(y_test, y_pred_logit_prod)
    auc_logit_prod = sklearn.metrics.roc_auc_score(y_test, y_pred_logit_prod)

    # measures the performance of logit model that uses the same input as the final model
    logit_classifier_all = logit_classifier.fit(
        X_train[all_features], np.ravel(y_train)
    )
    y_pred_logit = logit_classifier_all.predict_proba(X_test[all_features])[:, 1]
    bs_logit = sklearn.metrics.brier_score_loss(y_test, y_pred_logit)
    auc_logit = sklearn.metrics.roc_auc_score(y_test, y_pred_logit)

    # measures the performance of our lightGBM model
    lightgbm_classifier.fit(
        X_train[all_features], np.ravel(y_train)
    )
    y_pred_lgb = lightgbm_classifier_prod.predict_proba(X_test[all_features])[:, 1]
    bs_lgb = sklearn.metrics.brier_score_loss(y_test, y_pred_lgb)
    auc_lgb = sklearn.metrics.roc_auc_score(y_test, y_pred_lgb)

    # AUC scores and Brier scores
    # for the model and baselines are saved in yaml file
    scores = {
        "random": {
            "auc": float(auc_rnd),
            "brier_score": float(bs_rnd),
        },
        "logit_product_features_only": {
            "auc": float(auc_logit_prod),
            "brier_score": float(bs_logit_prod),
        },
        "logit": {
            "auc": float(auc_logit),
            "brier_score": float(bs_logit),
        },
        "lightgbm": {
            "auc": float(auc_lgb),
            "brier_score": float(bs_lgb),
        },
    }
    lib.write_yaml(scores, f"{path_results}/model_scores.yaml")
