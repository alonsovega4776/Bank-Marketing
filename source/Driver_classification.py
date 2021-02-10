"""
Alonso Vega
January 28, 2021
Driver: From here we will test our classifier.
"""
import ClassiferModel
import time

model = ClassiferModel.ClassifierModel()

# Naive Bayes
t_0 = time.time()
model.train_NB_parallelALL()
print("%s seconds" % (time.time() - t_0))

