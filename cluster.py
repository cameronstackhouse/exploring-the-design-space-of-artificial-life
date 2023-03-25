"""
Module containing experiments regarding the clustering of
both Xenobot behaviour and structure.
"""

import seaborn as sns
import tools.clustering
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

def clustering_goodness():
    """ 
    
    """
    pass

def cluster_behaviour():
    """ 
    
    """
    pass

def cluster_structure(xenobots):
    """ 
    
    """
    for xenobot in xenobots:
        pass

def classifier(dataframe: DataFrame):
    """ 
    Function to train an MLP classifier and 
    """
    X, y = dataframe.drop(["label"], axis=1), dataframe[["label"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = MLPClassifier(max_iter=300).fit(X_train, y_train)
    clf.predict(X_test)