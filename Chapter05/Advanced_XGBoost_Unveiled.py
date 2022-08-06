import pandas as pd
import numpy as np
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])
print(df.head())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=2)

xgb = XGBClassifier(booster='gbtree', objective='multi:softprob', 
                    learning_rate=0.1, n_estimators=100, random_state=2, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
score = accuracy_score(y_pred, y_test)
print('Score: ' + str(score))
