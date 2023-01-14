from sklearn.model_selection import GridSearchCV
import pca
import nb
import svm_func
import xgboost_model
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import utils

best_parameters = {
    "xgboost": {
        "learning_rate": [0.1],
        "max_depth": [5],
        "n_estimators": [100],
        "colsample_bytree": [0.5],
        "gamma": [1],
    }
}

tuning_parameters = {
    "xgboost": {
        "learning_rate": [0.1, 0.3, 0.5, 0.8],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "n_estimators": [100, 500],
        "colsample_bytree": [0.5, 0.8, 1],
        "gamma": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
}


def generate_tuned_model(model_name, attribute_training_data, class_training_data, pca_components,
                         use_best_model=False):
    model, modified_train_data = None, attribute_training_data
    if pca_components:
        modified_train_data = pca.get_modified_train_data(attribute_training_data, pca_components)

    if model_name == "naive_bayes" or model_name == "pca_naive_bayes":
        model = nb.generate_model()
    elif model_name == "svm" or model_name == "pca_svm":
        model = svm_func.generate_model()
    elif model_name == "xgboost" or model_name == "pca_xgboost":
        model = xgboost_model.generate_model()

    if use_best_model:
        gs = GridSearchCV(estimator=model, param_grid=best_parameters[model_name])
    else:
        gs = GridSearchCV(estimator=model, param_grid=tuning_parameters[model_name], verbose=10)

    gs.fit(modified_train_data, class_training_data)

    print("Best Parameters: ", gs.best_params_)
    return gs


def cost_function(model, X, y_true):
    y_pred = model.predict(X)
    conf = confusion_matrix(y_true, y_pred)
    cost = utils.get_cost(conf)
    print("cost function is working")
    return cost


custom_cost_scorer = make_scorer(cost_function, greater_is_better=True)
