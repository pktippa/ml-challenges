import numpy as np
# importing csv module to read the given input csv file
import csv
# Opening file in read mode
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
print_list = []
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

# SVM Linear kernel classification

C_values_for_svm_linear = [0.1, 0.5, 1, 5, 10, 50, 100]
tuned_parameters = [{'kernel': ['linear'], 'C': C_values_for_svm_linear}]
svm_lin_clf = svm.SVC()
svm_lin_grd_srch_clf = GridSearchCV(svm_lin_clf, tuned_parameters, cv=5, scoring='accuracy')
svm_lin_grd_srch_clf.fit(X_train, y_train)
svm_lin_train_scr = svm_lin_grd_srch_clf.best_score_
svm_lin_test_scr = accuracy_score(y_test, svm_lin_grd_srch_clf.predict(X_test))
print_list.append(['svm_linear', svm_lin_train_scr, svm_lin_test_scr])

# SVM Polynomial kernel classification

C_values_for_svm_poly = [0.1, 1, 3]
degree_vals_for_svm_poly = [4, 5, 6]
gamma_vals_for_svm_poly = [0.1, 0.5]
tuned_parameters = [{'kernel': ['poly'], 'C': C_values_for_svm_poly, 'degree': degree_vals_for_svm_poly, 'gamma': gamma_vals_for_svm_poly}]
svm_poly_clf = svm.SVC()
svm_poly_grd_srch_clf = GridSearchCV(svm_poly_clf, tuned_parameters, cv=5, scoring='accuracy')
svm_poly_grd_srch_clf.fit(X_train, y_train)
svm_poly_train_scr = svm_poly_grd_srch_clf.best_score_
svm_poly_test_scr = accuracy_score(y_test, svm_poly_grd_srch_clf.predict(X_test))
print_list.append(['svm_polynomial', svm_poly_train_scr, svm_poly_test_scr])

# SVM RBF(Radial Basis Function) kernel classification

C_values_for_svm_rbf = [0.1, 0.5, 1, 5, 10, 50, 100]
gamma_vals_for_svm_rbf = [0.1, 0.5, 1, 3, 6, 10]
tuned_parameters = [{'kernel': ['rbf'], 'C': C_values_for_svm_rbf,'gamma': gamma_vals_for_svm_rbf}]
svm_rbf_clf = svm.SVC()
svm_rbf_grd_srch_clf = GridSearchCV(svm_rbf_clf, tuned_parameters, cv=5, scoring='accuracy')
svm_rbf_grd_srch_clf.fit(X_train, y_train)
svm_rbf_train_scr = svm_rbf_grd_srch_clf.best_score_
svm_rbf_test_scr = accuracy_score(y_test, svm_rbf_grd_srch_clf.predict(X_test))
print_list.append(['svm_rbf', svm_rbf_train_scr, svm_rbf_test_scr])

# Logistic Regression

LR_best_scr = 0
C_values_for_L_R = [0.1, 0.5, 1, 5, 10, 50, 100]
LR_clf = LogisticRegressionCV(Cs=C_values_for_L_R, cv=5)
LR_clf.fit(X_train, y_train)
LR_train_scrs = LR_clf.scores_
for key,scoresList in LR_train_scrs.items():
    for scores in scoresList:
        LR_best_scr = max(scores) if max(scores) > LR_best_scr else LR_best_scr
LR_test_scr = accuracy_score(y_test, LR_clf.predict(X_test))
print_list.append(['logistic', LR_best_scr, LR_test_scr])

print(print_list)