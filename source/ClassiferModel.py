"""
Alonso Vega
January 23, 2021
ClassifierModel: Used to train a ensemble classifier.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
import operator

spark = SparkSession.builder\
        .master("local[4]")\
        .appName("parallel_classification")\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()


class ClassifierModel():
    __slots__ = '_X_mat', '_y_vect', '_freq_table', '_D_test', '_D_train'

    def __init__(self):
        try:
            self._X_mat  = pd.read_csv("X_matrix.csv")
            self._y_vect = pd.read_csv("y_vect.csv")
        except FileNotFoundError:
            print("File was not Found.")
            return None
        print("Data was read-in correctly.")

        self._D_test  = []
        self._D_train = []

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

    def split_data(self):
        current_data = pd.concat([self._X_mat, self._y_vect], axis=1)

        list_tt = train_test_split(current_data)
        D_train = list_tt[0]
        D_test  = list_tt[1]

        self._D_test  = D_test
        self._D_train = D_train
        return D_train

    def train_NB_parallelALL(self):
        feats = list(self._X_mat.columns)
        for feature in feats:
            self.train_NB_parallel(input_name=feature)

    def train_NB_parallel(self, input_name):
        # make RDD from each attr. and output
        data_df  = pd.concat([self._X_mat[input_name], self._y_vect], axis=1)
        data_df = spark.createDataFrame(data_df, schema=list(data_df.columns))

        data_rdd_0 = data_df.rdd.filter(lambda x: x["y"] == 0)
        data_rdd_0 = data_rdd_0.map(lambda x: x[input_name])

        data_rdd_1 = data_df.rdd.filter(lambda x: x["y"] == 1)
        data_rdd_1 = data_rdd_1.map(lambda x: x[input_name])

        unq_count_0 = data_rdd_0.map(lambda x: [x, 1]).sortByKey()
        unq_count_0 = unq_count_0.reduceByKey(operator.add)
        unq_count_1 = data_rdd_1.map(lambda x: [x, 1]).sortByKey()
        unq_count_1 = unq_count_1.reduceByKey(operator.add)

        count_df_0 = spark.createDataFrame(unq_count_0).toDF("Value", "Count").toPandas()
        count_df_1 = spark.createDataFrame(unq_count_1).toDF("Value", "Count").toPandas()

        if count_df_0.shape[0] != count_df_1.shape[0]:
            print("WARN: Count DF are not same size.")
            return None

        for i in range(count_df_0.shape[0]):
            self._freq_table[0][input_name][count_df_0.iloc[i, 0]] = count_df_0.iloc[i, 1]
        for i in range(count_df_1.shape[0]):
            self._freq_table[1][input_name][count_df_1.iloc[i, 0]] = count_df_1.iloc[i, 1]
        return None

    def train_NB(self):
        # Training
        D_train = self.split_data()
        #print("Training Matrix Size", D_train.shape[0], " x ", D_train.shape[1])
        #print(D_train.head(5))

        for i in range(D_train.shape[0]):
            inst       = D_train.iloc[i, :-1]
            this_class = D_train.iloc[i, -1]

            #print("\nInstance: ", i)
            for feat in inst.index:
                this_val = inst.loc[feat]
                self._freq_table[this_class][feat][this_val] += 1

                #print("Count of ({}, {}, {}) : ".format(this_class, feat, this_val),
                 #     self._freq_table[this_class][feat][this_val])
        return None

    def cond_prob_attr(self, attribute_name, class_name, attribute_value):
        # m-estimate
        m = 1

        # |X_i|
        attr_space = self._X_mat[attribute_name].unique()

        p_hat = self._freq_table[class_name][attribute_name][attribute_value]
        p_hat = p_hat + m*(1/attr_space.size)

        attr_expand = 0
        for val in attr_space:
            attr_expand = attr_expand + self._freq_table[class_name][attribute_name][val]

        p_hat = p_hat/(attr_expand + m)
        return p_hat

    def prob_class(self, class_name):
        train_output = self._D_train['y'].toNumpy()

        p_hat = (train_output == class_name).sum()
        p_hat = p_hat/len(train_output)

        return p_hat

    def test_NB(self):
        instance_test = self._D_test.sample(1)
        x_test        = instance_test.drop('y', axis=1)
        y_test        = instance_test['y'].to_numpy()[0]

        a_posteriori_dict = {}
        for class_i in np.unique(self._y_vect):
            class_prob = self.prob_class(class_i)

            cond_prob = 1
            for attr in x_test.column:
                cond_prob = cond_prob * self.cond_prob_attr(attr, class_i, x_test[attr])

            joint_prob = class_i*cond_prob
            a_posteriori_dict[class_i] = joint_prob

        a_posteriori_values = np.array(a_posteriori_dict.values())
        a_posteriori_max    = np.max(a_posteriori_values)
        y_hat               = a_posteriori_values[a_posteriori_max]

        return y_hat

    def train_SVM(self):
        return None

    def train_logistic_regression(self):
        return None

