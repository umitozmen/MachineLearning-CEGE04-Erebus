from Load_Dataset import readDatabaseFile
import numpy as np    
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn import tree
import graphviz

def read_data_return_frame(filename):
    
    df = readDatabaseFile(filename)
    
    #drop the columns that asuuming has less value for prediction
    df = df.drop(df.columns[[0,1,2,3,8,11,12,14]], axis=1)

    #get average of product realted and duration and drop product related visited page amount
    df['ProductRelatedAve'] = df.apply(lambda row: row.ProductRelated_Duration / row.ProductRelated if row.ProductRelated else 0, axis=1)
    new_coloumn_order = list(df.columns)
    new_coloumn_order.insert(2,'ProductRelatedAve')
    new_coloumn_order.pop()
    df = df.reindex(columns = new_coloumn_order)
    df = df.drop(df.columns[[0]], axis=1)

    return df

def preprocess_df(df):   
    

    #pre_processing month column into categorical attribute with one hot encoding
    # ohe = preprocessing.OneHotEncoder()
    # columnTransformer = ColumnTransformer([('encoder', ohe, [5,6])], remainder='passthrough')
    # df = np.array(columnTransformer.fit_transform(df), dtype = np.str)

    df['Month'] = pd.factorize(df.Month)[0]
    df['Region']  =  pd.factorize(df.Region)[0] 
    df['VisitorType'] = pd.factorize(df.VisitorType)[0]  
    df['Weekend'] = pd.factorize(df.Weekend)[0]  

    #return all column except last one for arttributes
    x = df.iloc[:,0: -1:1].values

    #return last column for label revenue true(1) / false(0)
    y = df.iloc[:, -1]
    y,class_names = pd.factorize(y)
    class_names = [str(x) for x in class_names]
    #get feature names
    feature_names = list(df.columns)[:-1]

    return x, y, class_names, feature_names

def train_test_classifier(x, y, test_size = 0.25, criterion='gini', max_depth =None):

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = test_size, stratify=y, random_state = 42)

    classifier = tree.DecisionTreeClassifier(criterion=criterion, random_state=42, max_depth= max_depth)
    classifier.fit(x_train, y_train)

    return x_train, x_test, y_train, y_test, classifier

def prediction(classifier, x):

    # Model prediction on given set
    y_pred =classifier.predict(x)

    return y_pred

def accuracy_cm_report(y, y_pred, class_names = []):

    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    print("Accuracy: {:.2f}".format(accuracy))

    cm=sklearn.metrics.confusion_matrix(y,y_pred)
    print('Confusion Matrix: \n', cm)
    print(sklearn.metrics.classification_report(y, y_pred, target_names=class_names))


def draw_tree(class_names, feature_names, classifier):

    dot_data = tree.export_graphviz(classifier, out_file=None, filled=True, rounded = True, feature_names=feature_names, class_names=class_names)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render('dtree_render',view=True)

def desicion_boundary(axis_0, axis_1, xs, ys):

    # training a decision tree only on two features
    xs = xs[:, [axis_0, axis_1]].astype(np.int)
    ys = ys.astype(np.int)

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # xs = sc.fit_transform(xs)

    tree_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    tree_clf.fit(xs[:, [axis_0, axis_1]], ys)

    # create a grid of points to plot the countour
    x_min, x_max = xs[:, axis_0].min() - 1, xs[:, axis_0].max() + 1
    y_min, y_max = xs[:, axis_1].min() - 1, xs[:, axis_1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # predict the outcome for the grid of points
    zz = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    cs = plt.contourf(xx, yy, zz)

    # # define axis
    # plt.xlabel(iris.feature_names[axis_0])
    # plt.ylabel(iris.feature_names[axis_1])

    # plot the dataset
    for i, color, marker in zip(range(2), 'ry', 'os'):
        idx = np.where(ys == i)
        plt.scatter(xs[idx, axis_0], xs[idx, axis_1], c=color, marker=marker, #label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    plt.show()

if __name__ == "__main__":

    data_frame_os = read_data_return_frame("online_shoppers_intention.csv")
    
    x, y, class_names, feature_names = preprocess_df(data_frame_os)

    print(class_names)
    #print(feature_names)
    #print(x[1:10,:])

    x_train, x_test, y_train, y_test, classifier = train_test_classifier(x, y, test_size = 0.25, criterion='gini', max_depth =5)

    y_pred_train = prediction(classifier, x_train)
    accuracy_cm_report(y_train, y_pred_train, class_names = class_names)

    #y_pred_test = prediction(classifier, x_test)
    #accuracy_cm_report(y_test, y_pred_test, class_names = class_names)

    #draw_tree(class_names, feature_names, classifier)

    #desicion_boundary(0,1,x_train,y_train)
