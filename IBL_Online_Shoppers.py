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
import math
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


def readDatabaseFile(filePath):
    # read csv data
    with open(filePath) as f:
        # load reads the csv db as a dictionary with
        # the data as a list of lists at key "data"
        dataFrame = pd.read_csv(f)
        f.close()

    return dataFrame


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

    # normalized_df.to_csv(r'C:\Users\prisc\Documents\\export_dataframe.csv', index=False, header=True)

    return x, y, class_names, feature_names


def preprocess_df_cat(data_frame_os_cat):
    # Separate input features and target
    y = data_frame_os_cat.Revenue
    y, class_names = pd.factorize(y)
    y = pd.DataFrame({'Revenue': y})
    x = data_frame_os_cat.drop('Revenue', axis=1)

    return x, y


def train_test(x, y, test_size=0.25):

    # setting up testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)

    return x_train, x_test, y_train, y_test


def upsample_minority(x, y):
    # concatenate our training data back together
    X = pd.concat([x, y], axis=1)

    # separate minority and majority classes
    not_true = X[X.Revenue == 0]
    true = X[X.Revenue == 1]

    # upsample minority
    true_upsampled = resample(true,
                              replace=True,  # sample with replacement
                              n_samples=len(not_true),  # match number in majority class
                              random_state=43)  # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_true, true_upsampled])

    # check new class counts
    upsampled.Revenue.value_counts()

    y = upsampled.Revenue
    x = upsampled.drop('Revenue', axis=1)

    return x, y


def downsample_majority(x, y):
    # concatenate our training data back together
    X = pd.concat([x, y], axis=1)

    # separate minority and majority classes
    not_true = X[X.Revenue == 0]
    true = X[X.Revenue == 1]

    # downsample minority
    not_true_downsampled = resample(not_true,
                                    replace=False,  # sample without replacement
                                    n_samples=len(true),  # match minority n
                                    random_state=42)  # reproducible results

    # combine majority and upsampled minority
    downsampled = pd.concat([not_true_downsampled, true])

    # check new class counts
    downsampled.Revenue.value_counts()

    y = downsampled.Revenue
    x = downsampled.drop('Revenue', axis=1)

    return x, y


def smote(x, y):

    sm = SMOTE(random_state=42)
    x_train, y_train = sm.fit_resample(x, y)
    # x_test, y_test = sm.fit_resample(x_test, y_test)
    print(y_train)
    # return x_train, y_train


def reshape(x, y):

    # concatenate our training data back together
    X = pd.concat([x, y], axis=1)

    # separate minority and majority classes
    not_true = X[X.Revenue == 0]
    true = X[X.Revenue == 1]

    # combine majority and upsampled minority
    reshape = pd.concat([not_true, true])

    y = reshape.Revenue
    x = reshape.drop('Revenue', axis=1)

    return x, y


def src_classifier_knn(x_train, y_train):
    # classifier = KNeighborsClassifier(n_neighbors=1, metric='cosine', weights='uniform')
    # k_min = int(k)-5
    # k_max = int(k)+5

    param_grid = [{
        'weights': ["uniform", "distance"],
        # 'n_neighbors': range(k_min,k_max),
        'n_neighbors': [1, 3, 5],
        'metric': ['euclidean', 'manhattan', 'cosine', 'minkowski', 'hamming']}]

    classifier = KNeighborsClassifier()
    grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose=0, scoring='f1')
    grid_search.fit(x_train, y_train)

    # parameters = {'kernel': ('linear', 'rbf', 'sigmoid'), 'gamma': ('scale', 'auto'), 'C': (7,8,13)}
    # svc = SVC()
    # grid_search = GridSearchCV(svc, parameters, cv=5, verbose=0, scoring='f1')
    # grid_search.fit(x_train, y_train)

    classifier = grid_search.best_estimator_
    print(classifier)
    print(grid_search.best_params_)

    classifier.fit(x_train, y_train)
    return classifier


def src_classifier_svm(x_train, y_train):
    # parameters = {'kernel': ('linear', 'rbf'), 'C': (15, 17, 19)}
    parameters = {'kernel': ('linear', 'rbf', 'sigmoid'), 'gamma': ('scale', 'auto'), 'C': (7, 8, 13)}
    svc = SVC()
    grid_search = GridSearchCV(svc, parameters, cv=5, verbose=0)
    grid_search.fit(x_train, y_train)

    classifier = grid_search.best_estimator_
    print(classifier)
    print(grid_search.best_params_)

    # classifier = SVC(kernel='sigmoid', gamma='scale', C=8)
    # print(classifier)

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
    f1score = sklearn.metrics.f1_score(y, y_pred)
    print("F1 Score: {:.2f}".format(f1score))

    print(f'{dom} Confusion matrix')
    cf = sklearn.metrics.confusion_matrix(y_pred, y)
    sns.heatmap(cf, annot=True, yticklabels=classes, xticklabels=classes, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()


def convert_num_to_cat(df):
    # SpecialDay column is actually split into 6 categories (0,0.2,0.4,0.6,0.8,1), so we just
    # multiply to get it to become an integer
    df['SpecialDay'] = 5 * df['SpecialDay']
    df['SpecialDay'] = df['SpecialDay'].astype('int64')

    # Product Related Duration, Product Related Average and Exit Rates binned into 5
    # categories (0-4) based on quantiles
    df['ProductRelated_Duration'] = pd.qcut(df['ProductRelated_Duration'], 5, labels=[0, 1, 2, 3, 4])
    df['ProductRelatedAve'] = pd.qcut(df['ProductRelatedAve'], 5, labels=[0, 1, 2, 3, 4])
    df['ExitRates'] = pd.qcut(df['ExitRates'], 5, labels=[0, 1, 2, 3, 4])

    # BounceRates binned into 3 categories based on quantiles (0-2) as there are
    # duplicate bins (of 0s) due to too many occurences
    df['BounceRates'] = pd.qcut(df['BounceRates'], 5, duplicates='drop', labels=[0, 1, 2])
    return df


if __name__ == "__main__":

    data_frame_os = read_data_return_frame("online_shoppers_intention.csv")
    x, y, class_names, feature_names = preprocess_df(data_frame_os)
    x_train, x_test, y_train, y_test = train_test(x, y, test_size=0.25)

    data_frame_os_cat = data_frame_os.copy()
    data_frame_os_cat = convert_num_to_cat(data_frame_os_cat)
    x_cat, y_cat = preprocess_df_cat(data_frame_os_cat)
    x_train, x_test, y_train, y_test = train_test(x_cat, y_cat, test_size=0.25)

    x_train, y_train = upsample_minority(x_train, y_train)
    # x_test, y_test = upsample_minority(x_test, y_test)

    # x_train, y_train = downsample_majority(x_train, y_train)
    # x_test, y_test = downsample_majority(x_test, y_test)

    # x_train, y_train = smote(x_train, y_train)

    # x_train, y_train = reshape(x_train, y_train)
    # x_test, y_test = reshape(x_test, y_test)

    # k = (math.sqrt(y_train.count()))
    # k = (math.sqrt(len(y_train)))

    classifier = src_classifier_svm(x_train, y_train)

    print("Accuracy Report for Training")
    y_pred_train = prediction(classifier, x_train)
    accuracy_cm_report(y_train, y_pred_train, class_names=class_names)

    print("\n")
    print("Accuracy Report for Testing")
    y_pred_test = prediction(classifier, x_test)
    accuracy_cm_report(y_test, y_pred_test, class_names=class_names)
