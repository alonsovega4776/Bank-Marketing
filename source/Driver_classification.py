"""
Alonso Vega
January 28, 2021
Driver: From here we will test our classifier.
"""
import ClassiferModel

model = ClassiferModel.ClassifierModel()

# Naive Bayes
model.train_NB()
model.test_NB()