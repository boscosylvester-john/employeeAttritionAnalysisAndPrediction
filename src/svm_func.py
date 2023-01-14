from sklearn import svm
from sklearn import metrics
import utils
import pca


def generate_model():
    print("\nGenerating SVM Model...")
    model = svm.SVC(kernel='linear')
    return model


def svm_do(attribute_training_data, class_training_data, pca_eval=False):
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


def generate_svm_model(attribute_training_data, class_training_data):
    model = generate_model()
    model.fit(attribute_training_data, class_training_data)
    return model


def generate_svm_with_PCA(attribute_training_data, class_training_data, pca_components):
    model = generate_model()
    return pca.generate_model_with_pca(model, pca_components, attribute_training_data, class_training_data)

#### TEST SCRIPT
# attribute_training_data, attribute_testing_data, class_training_data, class_testing_data = utils.get_data_split()
# svm_do(attribute_training_data, attribute_testing_data, class_training_data, class_testing_data)
