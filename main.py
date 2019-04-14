# 1.1 import necessary libraries
import numpy as np
import pandas as pd
import os
import pickle
import time
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pylab as plt

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, GridSearchCV
# from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from functions import *


# 1.2 load preprocessed data
train_file = 'input/oppo_round1_train_20180929.txt'
valid_file = './input/oppo_round1_vali_20180929.txt'
test_A_file = './input/oppo_round1_test_A_20180929.txt'

train_preprocessed_file = './preprocessed/train_preprocessed.pkl'
valid_preprocessed_file = './preprocessed/valid_preprocessed.pkl'
test_A_preprocessed_file = './preprocessed/test_A_preprocessed.pkl'

train_df, valid_df, test_df = load_data(train_file, train_preprocessed_file, valid_file, valid_preprocessed_file, test_A_file, test_A_preprocessed_file)


# 2. feature engineering

# items = ['prefix', 'prefix_len', 'pred_len', 'title','title_len', 'tag']
train_df, valid_df, test_df = feature_engineering(train_df, valid_df, test_df, items = ['prefix', 'title', 'tag'])


# 3. training model

X, y, X_valid, y_valid, X_test = split_data(train_df, valid_df, test_df)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 32,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}
model_path = './models/lgbm.pkl'
gbm = load_model(X, y, X_valid, y_valid, params, model_path)


# 3.3 plot feature importance 
fig_path = './images/feature_importance_lgbm.png'
plot_feature_importance(gbm, fig_path)

# 3.4 predict test data
time_now = time.strftime('%Y.%m.%d',time.localtime(time.time()))
submit_path = './submissions/gbm'+str(time_now)+'.csv'
model_predict(gbm, test_df, X_test, submit_path)