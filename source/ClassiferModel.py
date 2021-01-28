"""
Alonso Vega
January 23, 2021
ClassifierModel: Used to train a ensemble classifier.
"""
import pandas as pd
import numpy as np

class ClassifierModel():
    __slots__ = '_X_mat', '_y_vect', '_freq_table'

    def __init__(self):
        try:
            self._X_mat  = pd.read_csv("X_matrix.csv")
            self._y_vect = pd.read_csv("y_vect.csv")
        except FileNotFoundError:
            print("File was not Found.")
            return None
        print("Data was read-in correctly.")

        # Discretize
        self._X_mat["employees"] = pd.cut(self._X_mat["employees"], 25, labels=False)
        self._X_mat["euribor3Mon"] = pd.cut(self._X_mat["euribor3Mon"], 25, labels=False)
        self._X_mat["consConfidenceIdx"] = pd.cut(self._X_mat["consConfidenceIdx"], 25, labels=False)
        self._X_mat["consPriceIdx"] = pd.cut(self._X_mat["consPriceIdx"], 25, labels=False)
        self._X_mat["empVariation"] = pd.cut(self._X_mat["empVariation"], 25, labels=False)

        attr_list        = list(self._X_mat.columns)
        unqClasses_set   = list(np.unique(self._y_vect))
        self._freq_table = {unqClasses_set[i]: {} for i in range(len(unqClasses_set))}

        # Init. Table
        for i_class in range(len(unqClasses_set)):
            for attr in attr_list:
                self._freq_table[i_class][attr] = {}
                for val in self._X_mat[attr].unique():
                    self._freq_table[i_class][attr][val] = 0









    def train_selectiveNB(self):
        return None

    def train_SVM(self):
        return None

    def train_logistic_regression(self):
        return None

