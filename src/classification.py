import utils
import pca
import nb
import svm_func
import xgboost_model
import hyper_parameter_tuning
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def classify_for_best_model():
	attribute_training_data, attribute_testing_data, class_training_data, class_testing_data = utils.get_data_split()

	print("Finding best model...")
	model_name, pca_components = get_best_model(attribute_training_data, class_training_data)
	print("Optimal Model: ", model_name)
	if pca_components:
			print("Optimal PCA Components: ", pca_components)

	print("Training best model...")
	model, pca_model = generate_trained_model(model_name, attribute_training_data, class_training_data, pca_components)
	print("Training complete!")

	print("Testing...")
	if pca_model:
			pca_attribute_testing_data = transform_data_for_pca(pca_model, attribute_testing_data)
			generate_model_results(model, pca_attribute_testing_data, class_testing_data)
	else:
			generate_model_results(model, attribute_testing_data, class_testing_data)

	print("Tuning best model")
	model = hyper_parameter_tuning.generate_tuned_model(model_name, attribute_training_data, class_training_data, pca_components, True)
	print("Parameter tuning complete!")

	print("Testing best model after tuning")
	generate_model_results(model, attribute_testing_data, class_testing_data)


def get_best_model(attribute_training_data, class_training_data):
    available_models = {
        "naive_bayes": nb.naive_bayes(attribute_training_data, class_training_data, False),
        "svm": svm_func.svm_do(attribute_training_data, class_training_data, False),
        "xgboost": xgboost_model.xg_boost(attribute_training_data, class_training_data, False),
        # "pca_naive_bayes": nb.naive_bayes(attribute_training_data, class_training_data, True),
        "pca_svm": svm_func.svm_do(attribute_training_data, class_training_data, True),
        "pca_xgboost": xgboost_model.xg_boost(attribute_training_data, class_training_data, True),
    }
    print(available_models)
    # [best_model_name, pcaComponents] = max(available_models, key = lambda x: available_models[x][0])
    best_model, maxCost, optimalComp = None, float('-inf'), 0
    best_model_name = None
    for k, v in available_models.items():
        if v[0] > maxCost:
            best_model_name, maxCost, optimalComp = k, v[0], v[1]
    print("Best model: ", best_model_name, " | Cost: ", maxCost)
    return [best_model_name, optimalComp]


def generate_trained_model(model_name, attribute_training_data, class_training_data, pca_components=None):
    model, pcaModel = None, None
    if model_name == "naive_bayes":
        model = nb.generate_naive_bayes_model(attribute_training_data, class_training_data)
    elif model_name == "svm":
        model = svm_func.generate_svm_model(attribute_training_data, class_training_data)
    elif model_name == "xgboost":
        model = xgboost_model.generate_xgboost_model(attribute_training_data, class_training_data)
    elif model_name == "pca_naive_bayes":
        [model, pcaModel] = nb.generate_naive_bayes_with_PCA(attribute_training_data, class_training_data,
                                                             pca_components)
    elif model_name == "pca_svm":
        [model, pcaModel] = svm_func.generate_svm_with_PCA(attribute_training_data, class_training_data, pca_components)
    elif model_name == "pca_xgboost":
        [model, pcaModel] = xgboost_model.generate_xgboost_with_PCA(attribute_training_data, class_training_data,
                                                                    pca_components)

    return [model, pcaModel]


def transform_data_for_pca(pca_model, data):
    return pca_model.transform(data)


def generate_model_results(model, attribute_testing_data, class_testing_data):
    predicted_values = model.predict(attribute_testing_data)
    conf_matrix = confusion_matrix(class_testing_data, predicted_values)
    print("Confusion Matrix: \n", conf_matrix)

    print("Accuracy: ", accuracy_score(class_testing_data, predicted_values))

    print("Classification Report: \n", classification_report(class_testing_data, predicted_values))

    cost = utils.get_cost(conf_matrix)
    print("Final cost: ", cost)
