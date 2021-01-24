"""
Alonso Vega
January 23, 2021
ClassifierModel: Used to train a ensemble classifier.
"""

class ClassifierModel():
    __slots__ = '_X', '_y'

    def __int__(self, observationMatrix, outputs):
        self._X = observationMatrix
        self._y = outputs

    def train_selectiveNB(self):
        return None

    def train_SVM(self):
        return None

    def train_logistic_regression(self):
        return None

