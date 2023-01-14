from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn import metrics
import utils
import pca


def generate_model():
    print("\nGenerating Naive Bayes Model...")
    model = BernoulliNB()
    return model


def naive_bayes(attribute_training_data, class_training_data, pca_eval=False):
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


def generate_naive_bayes_model(attribute_training_data, class_training_data):
    model = generate_model()
    model.fit(attribute_training_data, class_training_data)
    return model


def generate_naive_bayes_with_PCA(attribute_training_data, class_training_data, pca_components):
    model = generate_model()
    return pca.generate_model_with_pca(model, pca_components, attribute_training_data, class_training_data)
