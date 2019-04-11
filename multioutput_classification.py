# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:45:47 2019

@author: SM
"""

import numpy as np
import pandas as pd

# create a dummy dataset of 5000 elements and 294 features, having multioutput target variable
X = np.random.random((5000, 294))
Y = np.random.randint(2, size=(5000, 6))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# create a random forest model using the default parameters, but n_estimators=300

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=0)

# create a multioutput classifier

from sklearn.multioutput import MultiOutputClassifier

multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)

Y_pred = multi_target_classifier.fit(X_train, Y_train).predict(X_test)

Y_pred_df = pd.DataFrame(Y_pred)

# calculate multioutput precision and loss
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss

avg_precision = label_ranking_average_precision_score(Y_test, Y_pred)
loss = label_ranking_loss(Y_test, Y_pred)

Y_pred_df.to_csv('Prediction.csv')