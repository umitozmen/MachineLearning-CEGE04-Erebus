# Modules needed for Dataset Exploration
import arff
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Modules needed for Machine Learning
import sklearn
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from statistics import *
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from collections import Counter
from numpy import sqrt
from numpy import argmax

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

    # get average of product realted and duration and drop product related visited page amount
    df['ProductRelatedAve'] = df.apply(
        lambda row: row.ProductRelated_Duration / row.ProductRelated if row.ProductRelated else 0, axis=1)
    new_coloumn_order = list(df.columns)
    new_coloumn_order.insert(2, 'ProductRelatedAve')
    new_coloumn_order.pop()
    df = df.reindex(columns=new_coloumn_order)
    df = df.drop(df.columns[[0]], axis=1)

    return df

def preprocess_df(df):

    #pre_processing month column into categorical attribute with one hot encoding
    # ohe = preprocessing.OneHotEncoder()
    # columnTransformer = ColumnTransformer([('encoder', ohe, [5,6])], remainder='passthrough')
    # df = np.array(columnTransformer.fit_transform(df), dtype = np.str)

    #factorize categorical attributes
    df['Month'] = pd.factorize(df.Month)[0]
    df['VisitorType'] = pd.factorize(df.VisitorType)[0]
    df['Weekend'] = pd.factorize(df.Weekend)[0]


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
