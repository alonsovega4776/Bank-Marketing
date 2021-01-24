"""
Alonso Vega
January 22, 2021
DataWrangler: This class will be used to prepossess our Data. We will also do EDA here.
                This class will prepare our data for classification modeling.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import autoimpute

class DataWrangler:
    __slots__ = '_data_df', '_dura_df', '_current_model_per_df'

    def __init__(self, csvFile_loc):

        if csvFile_loc.split(".")[-1] != "csv":
            print("ERROR: DataWrangler: Constructor: improper file location path.")
            print("Include full file name, with extension.")
            return None

        try:
            self._data_df = pd.read_csv(csvFile_loc)
        except FileNotFoundError:
            print("File was not Found.")
        print("Data was read-in correctly.")

        other_df = ['duration', 'ModelPrediction']

        self._dura_df              = self._data_df[other_df[0]]
        self._current_model_per_df = self._data_df[other_df[1]]

        self._data_df.drop(labels=other_df, axis='columns', inplace=True)

        data_names1 = list(self._data_df.columns)
        data_names2 = data_names1.copy()

        data_names2[5] = "houseLoan"
        data_names2[6] = "personalLoan"
        data_names2[7] = "commType"
        data_names2[8] = "priorMonth"
        data_names2[9] = "priorDay"

        data_names2[10] = "currentCampContacts"
        data_names2[11] = "priorCampDays"
        data_names2[12] = "priorCampContacts"
        data_names2[13] = "priorCampOutcome"

        data_names2[14] = "empVariation"
        data_names2[15] = "consPriceIdx"
        data_names2[16] = "consConfidenceIdx"
        data_names2[17] = "euribor3Mon"
        data_names2[18] = "employees"

        rename_dic = {data_names1[i]: data_names2[i] for i in range(len(data_names1))}
        self._data_df.rename(columns=rename_dic, inplace=True)

        print("Feature Space Dim: ", self._data_df.shape[0], 'x', self._data_df.shape[1])
        print(self._data_df.dtypes)

    def unique_observation(self, input_name, plot=False):
        num_bins = 10

        if input_name not in self._data_df:
            print("Feature: ", input_name, " not in Feature Space.")
            return None

        unq_values = self._data_df[input_name].unique()
        print("Unique Values of ", input_name, ": ")
        print(unq_values)

        if plot:
            hist_axes = sns.histplot(data=self._data_df, x=input_name, color=(0, 0.1, 0.5, 1),
                                     bins=num_bins, stat="count")
            plt.grid(True)
            plt.show()

        return unq_values

    def missing_data_stats(self, input_name):
        missing_data_df = self.subDF_obs(input_name=input_name, observation="unknown")
        print("Number of missing data instances for ", input_name, " :", missing_data_df.shape[0])





    def subDF_obs(self, input_name, observation):
        if input_name not in self._data_df:
            print("Feature: ", input_name, " not in Feature Space.")
            return None

        feat_col = self._data_df[input_name]
        unq_set  = self.unique_observation(input_name)

        if observation not in unq_set:
            print("Observation,", observation  ," ,is not present.")
            return 0
        else:
            observed_df = feat_col.loc[feat_col == observation]
            observed_df = self._data_df.loc[list(observed_df.index), :]
            return observed_df

    def fix_missing_data(self):
        # List-wise Deletion

        # Mean Imputation


        return None












