import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from CreateDataset import create_dataset
from sklearn.model_selection import train_test_split

dataset_path ="dataset"
x, y = create_dataset(dataset_path)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

f = open("MLT_result.txt", "w")

lr = LogisticRegression()

t0 = time()
#grid_values = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter':[5000, 10000], 'solver':['saga']}
# gscv = GridSearchCV(lr, param_grid=grid_values)
# gscv.fit(x_train, y_train)
# print("GridSearchCV() done in %0.3fs" % (time() - t0), file=f)
# print(gscv.cv_results_, file=f)
# print(gscv.best_params_, file=f)
# print(gscv.best_index_, file=f)
# print(gscv.best_score_, file=f)
# print(gscv.best_estimator_, file=f)

t0 = time()
#lr = LogisticRegression(**gscv.best_params_)
lr = LogisticRegression(max_iter =5000, solver='saga')
lr.fit(x_train, y_train)
print("LogisticRegression() done in %0.3fs" % (time() - t0), file=f)

y_pred = lr.predict(x_test)

print(classification_report(y_test, y_pred), file=f)
print(confusion_matrix(y_test, y_pred), file=f)

pickle.dump(lr, open("matrix.pkl", "wb"))

f.close()