import xgboost as xgb
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import root_numpy
import multiprocessing

from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone

# Data preparations

treename = 'tag'

data_b = pd.DataFrame(root_numpy.root2array('datasets/type=5.root', treename=treename)).dropna()
data_b = data_b[::20].reset_index(drop=True)
data_c = pd.DataFrame(root_numpy.root2array('datasets/type=4.root', treename=treename)).dropna().reset_index(drop=True)
data_light = pd.DataFrame(root_numpy.root2array('datasets/type=0.root', treename=treename)).dropna().reset_index(drop=True)
data_b['target'] = 0
data_c['target'] = 1
data_light['target'] = 2

data_b_c = pd.concat([data_b, data_c], ignore_index=True)
data_b_c = data_b_c.iloc[np.random.permutation(len(data_b_c))]

data_b_light = pd.concat([data_b, data_light], ignore_index=True)
data_b_light = data_b_light.iloc[np.random.permutation(len(data_b_light))]
data_b_light.loc[data_b_light['target'] == 2, 'target'] = 1

data_c_light = pd.concat([data_c, data_light], ignore_index=True)
data_c_light = data_c_light.iloc[np.random.permutation(len(data_c_light))]
data_c_light.loc[data_c_light['target'] == 1, 'target'] = 0
data_c_light.loc[data_c_light['target'] == 2, 'target'] = 1

data_bc_light = pd.concat([data_b, data_c, data_light], ignore_index=True)
data_bc_light = data_bc_light.iloc[np.random.permutation(len(data_bc_light))]
data_bc_light.loc[data_bc_light['target'] == 1, 'target'] = 0
data_bc_light.loc[data_bc_light['target'] == 2, 'target'] = 1

data_b_c_light = pd.concat([data_b, data_c, data_light], ignore_index=True)
data_b_c_light = data_b_c_light.iloc[np.random.permutation(len(data_b_c_light))]

sv_features = [column for column in data_b.columns if "SV" in column]

# Classifiers

#b_c 0.9501
xclf1 = xgb.XGBClassifier(learning_rate=0.02,
                         max_depth=6,
                         n_estimators=400,
                         gamma=2.0,
                         objective='binary:logistic',
                         colsample_bytree=0.85,
                         subsample=0.65,
                         min_child_weight=2)

#b_light 0.9867
xclf2 = xgb.XGBClassifier(learning_rate=0.02,
                         max_depth=6,
                         n_estimators=2000,
                         gamma=2,
                         objective='binary:logistic',
                         colsample_bytree=0.85,
                         subsample=0.65,
                         min_child_weight=2)

#c_light 0.9775
xclf3 = xgb.XGBClassifier(learning_rate=0.02,
                         max_depth=6,
                         n_estimators=2000,
                         gamma=2.5,
                         objective='binary:logistic',
                         colsample_bytree=0.85,
                         subsample=0.65,
                         min_child_weight=2)

#b_c_light 0.9501 0.9865 0.9777
xclf4 = xgb.XGBClassifier(learning_rate=0.02,
                         max_depth=6,
                         n_estimators=2000,
                         gamma=2.5,
                         objective='binary:logistic',
                         colsample_bytree=0.85,
                         subsample=0.65,
                         min_child_weight=2)

def compute_multiclass_auc(data):
    train = data[0]
    test = data[1]
    clf = clone(data[2])
    X_train = train.drop('target', axis=1).values
    y_train = train['target'].values

    X_test = test.drop('target', axis=1).values
    y_test = test['target'].values

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    return [1 - roc_auc_score(y_test > 0, y_pred[:, 0] / y_pred[:, 1],
                                                      sample_weight=(y_test != 2) * 1),
    1 - roc_auc_score(y_test > 1, y_pred[:, 0] / y_pred[:, 2],
                                                      sample_weight=(y_test != 1) * 1),
    1 - roc_auc_score(y_test > 1, y_pred[:, 1] / y_pred[:, 2],
                                                      sample_weight=(y_test != 0) * 1)]

def compute_auc(data):
    train = data[0]
    test = data[1]
    clf = clone(data[2])
    X_train = train.drop('target', axis=1).values
    y_train = train['target'].values

    X_test = test.drop('target', axis=1).values
    y_test = test['target'].values

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    return roc_auc_score(y_test, y_pred[:, 1])

# Calculations

'''dataset = data_b_c_light
estimator = compute_multiclass_auc
clf = xclf4

'''
dataset = data_b_c
estimator = compute_auc
clf = xclf1

foldation = KFold(dataset.shape[0], 2, shuffle=True, random_state=0)
pool = multiprocessing.Pool(4)
output = pool.map(estimator, [(dataset[sv_features + ['target']].iloc[train_index], dataset[sv_features + ['target']].iloc[test_index], clf)
    for train_index, test_index in foldation])
pool.close()

print(np.mean(output, axis=0))