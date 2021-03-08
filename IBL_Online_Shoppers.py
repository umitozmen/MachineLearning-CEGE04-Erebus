from Load_Dataset import readDatabaseFile
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample


def read_data_return_frame(filename):
    df = readDatabaseFile(filename)

    # drop the columns that asuuming has less value for prediction
    df = df.drop(df.columns[[0, 1, 2, 3, 8, 11, 12, 14]], axis=1)

    # get average of product related and duration and drop product related visited page amount
    df['ProductRelatedAve'] = df.apply(
        lambda row: row.ProductRelated_Duration / row.ProductRelated if row.ProductRelated else 0, axis=1)
    new_coloumn_order = list(df.columns)
    new_coloumn_order.insert(2, 'ProductRelatedAve')
    new_coloumn_order.pop()
    df = df.reindex(columns=new_coloumn_order)
    df = df.drop(df.columns[[0]], axis=1)

    return df


def preprocess_df(df):
    # pre_processing month column into categorical attribute with one hot encoding
    # ohe = preprocessing.OneHotEncoder()
    # columnTransformer = ColumnTransformer([('encoder', ohe, [5,6])], remainder='passthrough')
    # df = np.array(columnTransformer.fit_transform(df), dtype = np.str)

    # factorize categorical attributes
    df['Month'] = pd.factorize(df.Month)[0]
    df['Region'] = pd.factorize(df.Region)[0]
    df['VisitorType'] = pd.factorize(df.VisitorType)[0]
    df['Weekend'] = pd.factorize(df.Weekend)[0]

    # return all column except last one for arttributes
    x = df.iloc[:, 0: -1:1].values

    # return last column for label revenue true(1) / false(0)
    y = df.iloc[:, -1]
    y, class_names = pd.factorize(y)
    class_names = [str(x) for x in class_names]

    # get feature names
    feature_names = list(df.columns)[:-1]

    return x, y, class_names, feature_names


def train_test(x, y, test_size=0.25):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)
    return x_train, x_test, y_train, y_test


def resample_data(x_train, y_train):
    # concatenate our training data back together


    for i in range(len(x_train)):
        # separate minority and majority classes
        if y_train[i] == 0:
            not_revenue.append(X_Y[i])
        else:
            revenue.append(X_Y[i])

    # up-sample minority
    revenue_upsampled = resample(revenue,
                                 replace=True,  # sample with replacement
                                 n_samples=len(not_revenue),  # match number in majority class
                                 random_state=42)  # reproducible results


    # # combine majority and upsampled minority
    # upsampled = pd.concat([not_revenue, revenue_upsampled])
    #
    # # check new class counts
    # print("check new class counts")
    # print(upsampled.Class.value_counts())
    #
    # y_resample = upsampled.Class
    # x_resample = upsampled.drop('Class', axis=1)
    #
    # return x_resample, y_resample


def src_category(x_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    # param_grid = [{
    #     'weights': ["uniform", "distance"],
    #     'n_neighbors': range(1, 5),
    #     'metric': ['cosine']}]
    # # 'metric': ['euclidean', 'manhattan', 'cosine']}]

    # classifier = KNeighborsClassifier(metric="manhattan", n_neighbors=9, weights="distance")

    # classifier = KNeighborsClassifier()
    # grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose=2)
    # grid_search.fit(x_train, y_train)
    #
    # classifier = grid_search.best_estimator_
    print(classifier)

    classifier.fit(x_train, y_train)
    return classifier


def prediction(classifier, x):
    # Model prediction on given set
    y_pred = classifier.predict(x)

    return y_pred


def accuracy_cm_report(y, y_pred, class_names=[], dom='Train', brk=True):
    print(f'{dom} score {sklearn.metrics.accuracy_score(y, y_pred)}')

    if brk:
        plot_confusionmatrix(y_pred, y, class_names, dom=dom)

    report = sklearn.metrics.classification_report(y, y_pred, target_names=class_names, output_dict=True)
    cls_report_df = pd.DataFrame(report)

    print(cls_report_df.iloc[0:2, 0:2])


def plot_confusionmatrix(y_pred, y, classes, dom):
    print(f'{dom} Confusion matrix')
    cf = sklearn.metrics.confusion_matrix(y_pred, y)
    sns.heatmap(cf, annot=True, yticklabels=classes, xticklabels=classes, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    data_frame_os = read_data_return_frame("online_shoppers_intention.csv")

    x, y, class_names, feature_names = preprocess_df(data_frame_os)
    x_train, x_test, y_train, y_test = train_test(x, y, test_size=0.25)

    resample_data(x_train, y_train)
    # classifier = src_category(x_train, y_train)

    # print("Accuracy Report for Training")
    # y_pred_train = prediction(classifier, x_train)
    # accuracy_cm_report(y_train, y_pred_train, class_names=class_names)

    # print("Accuracy Report for Testing")
    # y_pred_test = prediction(classifier, x_test)
    # accuracy_cm_report(y_test, y_pred_test, class_names=class_names)
