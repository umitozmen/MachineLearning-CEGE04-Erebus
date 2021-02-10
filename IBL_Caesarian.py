from Load_Ceserian_Dataset import readCeserianFile
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def read_data_return_frame(filename):
    dataframe = readCeserianFile(filename)
    feature_names = list(dataframe.columns)[:-1]

    #   return all column except last one for arttributes
    x = dataframe.iloc[:, 0: -1:1].values

    #   return last column for label ceserian yes(1) / no(0)
    y = dataframe.iloc[:, -1]
    y, class_names = pd.factorize(y)

    return x, y, class_names, feature_names


# def train_test_classifier(x, y, test_size=0.25):
#
#     x_train, x_test, y_train, y_test = \
#         model_selection.train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)
#
#     #   classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
#     param_grid = [{
#         'weights': ["uniform", "distance"],
#         'n_neighbors': range(1, 11),
#         'metric': ['euclidean', 'manhattan', 'cosine']}]
#
#     # classifier = KNeighborsClassifier(metric="manhattan", n_neighbors=9, weights="distance")
#
#     classifier = KNeighborsClassifier()
#     grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose=2)
#     grid_search.fit(x_train, y_train)
#
#     classifier = grid_search.best_estimator_
#     print(classifier)
#     classifier.fit(x_train, y_train)
#     return x_train, x_test, y_train, y_test, classifier


def train_test_classifier(x, y, test_size=0.25):

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = SVC()

    grid_search= GridSearchCV(svc, parameters)
    grid_search.fit(x_train, y_train)

    classifier = grid_search.best_estimator_
    print(classifier)
    classifier.fit(x_train, y_train)
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
    print('Confusion Matrix: \n', cm)
    print(sklearn.metrics.classification_report(y, y_pred, target_names=class_names))


if __name__ == "__main__":

    x, y, class_names, feature_names = read_data_return_frame("caesarian.csv.arff")

    x_train, x_test, y_train, y_test, classifier = train_test_classifier(x, y, test_size=0.25)

    print("Accuracy Report for Training")
    y_pred_train = prediction(classifier, x_train)
    accuracy_cm_report(y_train, y_pred_train, class_names=class_names)

    print("Accuracy Report for Testing")
    y_pred_test = prediction(classifier, x_test)
    accuracy_cm_report(y_test, y_pred_test, class_names=class_names)
