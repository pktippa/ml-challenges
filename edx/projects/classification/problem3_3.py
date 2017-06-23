import numpy as np
# importing csv module to read the given input csv file
import csv
# Opening file in read mode
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
input_csv=open("input3.csv","r")
# Getting the input csv text content
input_csv_text=csv.reader(input_csv)
# Removes the header.
next(input_csv_text)
input_data = list([float(el[0]), float(el[1]), int(el[2])] for el in input_csv_text)
input_data_arr = np.array(input_data)
y_label=input_data_arr[:,2]
X_features = np.delete(input_data_arr, 2, axis=1)
y_label = y_label.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, train_size=0.6, stratify=y_label)
C_values_for_svm_linear = [0.1, 0.5, 1, 5, 10, 50, 100]
for c in C_values_for_svm_linear:
    print(c)
    svr_lin_clf = svm.SVC(kernel='linear', C=c)
    scores = cross_val_score(svr_lin_clf, X_train, y_train, cv=5)
    print(scores)