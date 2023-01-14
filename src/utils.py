# -*- coding: utf-8 -*-
import pandas as pd
import math as m
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import KFold

### Loading Data
raw_data_file = "data/watson_healthcare_modified.csv"
normalized_data_file = "data/normalized_data.csv"
preproc_data_file = "data/preprocessed_data.csv"

### No. of splits
splits = 10


improvable_features = {
    "DailyRate": {"stdev": None, "mean": None},
    "HourlyRate": {"stdev": None, "mean": None},
    "MonthlyIncome": {"stdev": None, "mean": None},
    "MonthlyRate": {"stdev": None, "mean": None},
    "PercentSalaryHike": {"stdev": None, "mean": None},
    "TrainingTimesLastYear": {"stdev": None, "mean": None},
    "DistanceFromHome": {"stdev": None, "mean": None},
    "YearsInCurrentRole": {"stdev": None, "mean": None},
    "YearsSinceLastPromotion": {"stdev": None, "mean": None},
    "YearsWithCurrManager": {"stdev": None, "mean": None}
}


def generate_preprocessed_dataset():
    original_data = pd.read_csv(raw_data_file)
    cols = original_data.columns
    trimmed_data = original_data.drop(columns=['Over18', 'EmployeeCount', 'StandardHours', 'PerformanceRating'])

    label_encoder = preprocessing.LabelEncoder()
    categorical_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender',
                           'JobRole', 'MaritalStatus', 'OverTime', 'Education', 'JobLevel', 'Shift']
    for i in categorical_columns:
        trimmed_data[i] = label_encoder.fit_transform(trimmed_data[i])

    trimmed_data = trimmed_data.fillna(0)

    sc = preprocessing.StandardScaler()
    scaled_columns = list(
        set(trimmed_data.columns).difference(set(categorical_columns + ['EmployeeID']))
    )
    for col in scaled_columns:
        if col in improvable_features:
            improvable_features[col]["stdev"] = trimmed_data[col].std()
            improvable_features[col]["mean"] = trimmed_data[col].mean()

    continuous_data = trimmed_data[scaled_columns]
    trimmed_data[scaled_columns] = sc.fit_transform(continuous_data)

    trimmed_data.to_csv(preproc_data_file, encoding='utf-8', index=False)


def get_preprocessed_dataset():
    preproc_df = pd.read_csv(preproc_data_file)
    return preproc_df


def get_improvable_features():
    return improvable_features


def generate_normalized_dataset():
    original_data = pd.read_csv(raw_data_file)
    cols = original_data.columns

    continuous_columns = []
    for i in cols:
        num_vals = len(set(original_data[i]))
        if num_vals >= 20:
            continuous_columns.append(i)

    label_encoder = preprocessing.LabelEncoder()
    categorical_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender',
                           'JobRole', 'MaritalStatus', 'OverTime']
    for i in categorical_columns:
        original_data[i] = label_encoder.fit_transform(original_data[i])

    trimmed_data = original_data.drop(columns=['EmployeeID', 'Over18', 'EmployeeCount', 'StandardHours'])

    cols = trimmed_data.columns
    for i in cols:
        if i not in categorical_columns and i in continuous_columns:
            mx = max(trimmed_data[i])
            mn = min(trimmed_data[i])
            m = (len(str(mx)) - 1) * 10
            binlist = [i for i in range(mn, mx, m)] + [int(mx)]
            trimmed_data[i] = pd.cut(trimmed_data[i], bins=binlist, labels=False, right=True) + 1

    filled_data = trimmed_data.fillna(0)

    continuous_columns = []
    for i in cols:
        num_vals = len(set(filled_data[i]))
        if num_vals > 20:
            continuous_columns.append(i)

    normalized_data = filled_data
    for i in continuous_columns:
        d = preprocessing.normalize([np.array(normalized_data[i])])
        normalized_data.drop(i, axis=1, inplace=True)
        normalized_data[i] = d[0]

    normalized_data.to_csv(normalized_data_file, encoding='utf-8')


def get_normalized_dataset():
    normalized_df = pd.read_csv(normalized_data_file)
    return normalized_df


def get_data_split():
    normalized_data = pd.read_csv(normalized_data_file)
    y_label_name = "Attrition"

    y = normalized_data[y_label_name]

    x = normalized_data.loc[:, normalized_data.columns != y_label_name]
    x = x.iloc[:, 1:]

    # getting data for train, validation and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    # x_train, x_val, y_train, y_val = train_test_split(x_trn, y_trn, test_size=0.25, random_state=1)

    return x_train, x_test, y_train, y_test


# model_kfold is new instance of the model, for eg: svm.SVC()
def generate_confusion_matrix(model_kfold, X, Y):
    cv = KFold(n_splits=splits)
    y_pred = cross_val_predict(model_kfold, X, Y, cv=cv)
    confusion_matrix_kfold = confusion_matrix(Y, y_pred)
    return confusion_matrix_kfold


def get_cost(confusion_matrix_kfold):
    cost_matrix = [[10, -100], [-25, 150]]
    cost = 0
    for i in range(2):
        for j in range(2):
            cost += cost_matrix[i][j] * confusion_matrix_kfold[i][j]
    return cost


def kfoldcv(model_kfold, X, Y, pca_comp=-1):
    matrix = generate_confusion_matrix(model_kfold, X, Y)
    return get_cost(matrix)
