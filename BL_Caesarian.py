import arff
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from mixed_naive_bayes import MixedNB
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from matplotlib import pyplot as plt

def readCeserianFile(filePath):
    # read arff data
    with open(filePath) as f:
        # load reads the arff db as a dictionary with
        # the data as a list of lists at key "data"
        dataDictionary = arff.load(f)
        f.close()

    # extract data and convert to numpy array
    arffData = np.array(dataDictionary['data'])
    arffAttributes = [i[0] for i in dataDictionary['attributes']]

    return pd.DataFrame(arffData, columns=arffAttributes)


def read_data_return_frame(filename):
    dataframe = readCeserianFile(filename)
    feature_names = list(dataframe.columns)[:-1]

    #   return all column except last one for arttributes
    x = dataframe.iloc[:, 0: -1:1].values

    #   return last column for label ceserian yes(1) / no(0)
    y = dataframe.iloc[:, -1]
    y, class_names = pd.factorize(y)

    return x, y, class_names, feature_names


def train_test_classifier(x, y, test_size=0.25, classifier="Multinomial"):
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)

    param_grid = [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 5, 10]}]

    if classifier in ('Gaussian', 'Categorical'):

        if classifier == 'Gaussian':
            classifier = GaussianNB()
            classifier.fit(x_train, y_train)
            print('Classes: ', classifier.classes_)
            print('Class Priors: ', classifier.class_prior_)
        elif classifier == "Categorical":
            classifier = CategoricalNB()
            classifier.fit(x_train, y_train)
            print('Classes: ', classifier.classes_)
            print('Class Log Priors: ', classifier.class_log_prior_)
    else:
        if classifier == "Multinomial":
            classifier = MultinomialNB()
        elif classifier == "Complement":
            classifier = ComplementNB()
        elif classifier == "Bernoulli":
            classifier = BernoulliNB()

        grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose=2)
        grid_search.fit(x_train, y_train)
        classifier = grid_search.best_estimator_
        print(classifier)
        classifier.fit(x_train, y_train)
        print('Classes: ', classifier.classes_)
        print('Class Log Priors: ', classifier.class_log_prior_)
    return x_train, x_test, y_train, y_test, classifier


def prediction(classifier, x):
    # Model prediction on given set
    # y_pred = classifier.predict(x)
    y_pred = classifier.predict(x)
    return y_pred


def accuracy_cm_report(y, y_pred, class_names=[]):
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    print("Accuracy: {:.2f}".format(accuracy))

    cm = sklearn.metrics.confusion_matrix(y, y_pred)
    print(sklearn.metrics.classification_report(y, y_pred, target_names=class_names))

    print('Confusion Matrix: \n', cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="BuPu",
                ax=ax)
    plt.xlabel('true label')
    plt.ylabel('predicted label')


if __name__ == "__main__":
    df = readCeserianFile("caesarian.csv.arff")
    x, y, class_names, feature_names = read_data_return_frame("caesarian.csv.arff")


