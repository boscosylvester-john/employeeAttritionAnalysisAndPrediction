# -*- coding: utf-8 -*-
"""xgboost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q3afZB8EaACysSCvhK4mAS-NxGHFgv8j
"""

import xgboost as xgb
import pca
import utils
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def generate_model():
    print("\nGenerating XGBoost Model...")
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    return model


def xg_boost(attribute_training_data, class_training_data, pca_eval=False):
    model = generate_model()
    if pca_eval:
        print("Applying PCA...")
        [modelcost, optimalComp] = pca.getOptimalPCAComponents(model, attribute_training_data, class_training_data)
        print("Model cost: ", modelcost, " | Optimal number of principal components: ", optimalComp)
        return [modelcost, optimalComp]
    else:
        modelcost = utils.kfoldcv(model, attribute_training_data, class_training_data)
        print("Model cost: ", modelcost)
        return [modelcost, None]


def generate_xgboost_model(attribute_training_data, class_training_data):
    model = generate_model()
    model.fit(attribute_training_data, class_training_data)
    return model


def generate_xgboost_with_PCA(attribute_training_data, class_training_data, pca_components):
    model = generate_model()
    return pca.generate_model_with_pca(model, pca_components, attribute_training_data, class_training_data)
