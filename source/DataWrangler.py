"""
Alonso Vega
January 22, 2021
DataWrangler: This class will be used to prepare our Data for modeling. We will also do EDA here.
                This class will prepare our data for classification modeling.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import autoimpute
import numpy as np
import calendar
import sklearn.preprocessing as skl_pre


pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 22)
sns.set_theme()

class DataWrangler:
    __slots__ = '_feature_df', '_output_df',\
                '_dura_df', '_current_model_per_df'

    def __init__(self, csvFile_loc):

        if csvFile_loc.split(".")[-1] != "csv":
            print("ERROR: DataWrangler: Constructor: improper file location path.")
            print("Include full file name, with extension.")
            return None

        try:
            self._feature_df = pd.read_csv(csvFile_loc)
        except FileNotFoundError:
            print("File was not Found.")
        print("Data was read-in correctly.")

        other_df = ['y', 'duration', 'ModelPrediction']

        self._output_df            = self._feature_df[other_df[0]]
        self._dura_df              = self._feature_df[other_df[1]]
        self._current_model_per_df = self._feature_df[other_df[2]]

        self._feature_df.drop(labels=other_df, axis='columns', inplace=True)
        print("Feature Space Dim: ", self._feature_df.shape[0], 'x', self._feature_df.shape[1])

        data_names1 = list(self._feature_df.columns)
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
        self._feature_df.rename(columns=rename_dic, inplace=True)

        self.deep_cleaning()

        print(self._feature_df.dtypes)

    def unique_observation(self, input_name, plot=False):
        num_bins = 10

        if input_name not in self._feature_df:
            print("Feature: ", input_name, " not in Feature Space.")
            return None

        unq_values = self._feature_df[input_name].unique()

        if plot:
            hist_axes = sns.histplot(data=self._feature_df, x=input_name, color=(0, 0.1, 0.5, 1),
                                     bins=num_bins, stat="count")
            plt.grid(True)
            plt.show()

        return unq_values

    def subDF_obs(self, input_name, observation):
        if input_name not in self._feature_df:
            print("Feature: ", input_name, " not in Feature Space.")
            return None

        feat_col = self._feature_df[input_name]
        unq_set  = self.unique_observation(input_name)

        if observation not in unq_set:
            print("Observation,", observation  ," ,is not present.")
            return 0
        else:
            observed_df = feat_col.loc[feat_col == observation]
            observed_df = self._feature_df.loc[list(observed_df.index), :]
            return observed_df

    def deep_cleaning(self):
        # Duplicates
        num_clones = self._feature_df.duplicated()
        num_clones = num_clones.sum()
        print("Number of duplicates in feature+output matrix: ", num_clones)

        org_data = pd.concat([self._feature_df, self._output_df, self._dura_df, self._current_model_per_df], axis=1)
        dup_col_list = list(self._feature_df.columns)
        dup_col_list.append(list(self._output_df.name)[0])
        org_data.drop_duplicates(inplace=True, subset=dup_col_list)

        self._output_df            = org_data["y"]
        self._dura_df              = org_data["duration"]
        self._current_model_per_df = org_data["ModelPrediction"]
        self._feature_df           = org_data.iloc[:, :-3]

        print("Feature matrix dim after duplicate deletion: ", self._feature_df.shape[0], ' x ', self._feature_df.shape[1])

        # Numeric Preference
        replace_dic = {"yes": 1, "no": 0}
        self._output_df.replace(to_replace=replace_dic, inplace=True)

        replace_dic["unknown"] = 0.5
        binary_features = ["default", "personalLoan", "houseLoan"]
        for x in binary_features:
            self._feature_df[x].replace(to_replace=replace_dic, inplace=True)

        days        = ['mon', 'tue', 'wed', 'thu', 'fri']
        daysNum     = np.arange(1, 6, 1)
        replace_dic = {days[i]: daysNum[i] for i in range(len(days))}
        self._feature_df["priorDay"].replace(to_replace=replace_dic, inplace=True)

        months      = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        monthsNum   = np.arange(1, 13, 1)
        replace_dic = {months[i]: monthsNum[i] for i in range(len(months))}
        self._feature_df["priorMonth"].replace(to_replace=replace_dic, inplace=True)

    def feature_description(self, input_name):
        if input_name not in list(self._feature_df.columns):
            print("\nFeature: ", input_name, " not in Feature Space.")
            return None

        feat_col = self._feature_df[input_name]

        pph = np.arange(0.15, 1.0, 0.15)
        info = feat_col.describe(percentiles=pph)
        print("\n----------------", input_name, "----------------")
        unq_values = self.unique_observation(input_name, plot=False)
        print("--> Unique Values : ")
        print(unq_values)
        print("Size of unique set: ", len(unq_values), "\n")

        count_info = feat_col.value_counts(sort=True)
        print("--> Value count (Sorted): ")
        print(count_info)

        print("Value count (Normalized): ")
        count_info = feat_col.value_counts(normalize=True, sort=True)
        print(count_info)

        print("\n--> Stats:")
        print(info)

        print("\n--> Pivot Table: ")
        feat_col_output = pd.concat([feat_col, self._output_df], axis=1)
        pivot_table = pd.pivot_table(data=feat_col_output,
                                     values="y", index=input_name,
                                     aggfunc=[np.mean, np.std, np.var])
        pivot_table.sort_values(by=pivot_table.columns[0], inplace=True, ascending=False)
        print(pivot_table)
        print("----------------", input_name, "----------------")

    def prior_campaign(self, plot=False):
        prior_camp_df = self._feature_df.iloc[:, 11:14]
        print("\n----------------Previous Campaign Info---------------- \n",
              prior_camp_df.head())
        print("\n-->Random sample:")
        print(prior_camp_df.sample(n=20, axis=0))

        # Contradictory Data
        print("\n-->Contradiction: ")
        odd_df_1 = prior_camp_df[(prior_camp_df["priorCampDays"] == 999) & (prior_camp_df["priorCampContacts"] != 0)]
        print("\n1.)Odd:\n", odd_df_1.head())
        odd_df_2 = prior_camp_df[(prior_camp_df["priorCampDays"] != 999) & (prior_camp_df["priorCampContacts"] == 0)]
        print("\n2.)Odd:\n", odd_df_2.head())

        odd_df_3 = prior_camp_df[(prior_camp_df["priorCampDays"] != 999) & (prior_camp_df["priorCampOutcome"] == "nonexistent")]
        print("\n3.)Odd:\n", odd_df_3.head())
        odd_df_4 = prior_camp_df[(prior_camp_df["priorCampDays"] == 999) & (prior_camp_df["priorCampOutcome"] != "nonexistent")]
        print("\n4.)Odd:\n", odd_df_4.head())

        odd_df_5 = prior_camp_df[(prior_camp_df["priorCampContacts"] != 0) & (prior_camp_df["priorCampOutcome"] == "nonexistent")]
        print("\n5.)Odd:\n", odd_df_5.head())
        odd_df_6 = prior_camp_df[(prior_camp_df["priorCampContacts"] == 0) & (prior_camp_df["priorCampOutcome"] != "nonexistent")]
        print("\n6.)Odd:\n", odd_df_6.head())

        # Delete This Data
        print("\n-->Deleting odd observation.")
        '''
        current_data = pd.concat([self._feature_df, self._output_df, self._dura_df, self._current_model_per_df], axis=1)
        current_data.drop(labels=odd_df_1.index, axis=0, inplace=True)

        self._output_df            = current_data["y"]
        self._dura_df              = current_data["duration"]
        self._current_model_per_df = current_data["ModelPrediction"]
        self._feature_df           = current_data.iloc[:, :-3]
        '''
        current_data = self.del_examples(odd_df_1.index)

        print("New Feature Space Dim: ", self._feature_df.shape[0], 'x', self._feature_df.shape[1])
        print("Number of data deleted: ", odd_df_1.shape[0])

        # History on previous campaign
        prior_camp_df = prior_camp_df[prior_camp_df["priorCampDays"] != 999]

        pivot_table_days     = pd.pivot_table(data=prior_camp_df,
                                          index="priorCampOutcome", values="priorCampDays",
                                          aggfunc=[np.mean, np.std, np.max, np.min])
        pivot_table_contacts = pd.pivot_table(data=prior_camp_df,
                                          index="priorCampOutcome", values="priorCampContacts",
                                          aggfunc=[np.mean, np.std, np.max, np.min])
        print("\n-->Previous Campaign Stats: ")
        print(pivot_table_days)
        print(pivot_table_contacts)

        # New customers info
        current_data     = current_data.iloc[:, :-2]
        old_customers_df = current_data.loc[list(prior_camp_df.index), :]
        current_data.drop(list(prior_camp_df.index), axis=0, inplace=True)
        new_customers_df = current_data

        print("\n-->New Customers")
        print("Number of new clients: ", new_customers_df.shape[0])
        print("Number of old clients: ", old_customers_df.shape[0])

        # Graph success outcome wrt last month contacted
        month_count = [old_customers_df["priorMonth"].value_counts().sort_index(),
                       new_customers_df["priorMonth"].value_counts().sort_index()]

        pos_outcomesPT = [pd.pivot_table(old_customers_df, values='y', index="priorMonth", aggfunc=np.sum).iloc[:, 0],
                          pd.pivot_table(new_customers_df, values='y', index="priorMonth", aggfunc=np.sum).iloc[:, 0]]

        success_freq = []
        i            = 0
        for people in pos_outcomesPT:
            success_freq.append(people/month_count[i])
            i = i + 1

        success_freq = pd.concat(success_freq, axis=1)
        success_freq.rename(columns={0: "Old Clients", 1: "New Clients"}, inplace=True)

        ax = sns.lineplot(data=success_freq, markers=['o', 'o'], dashes=False, palette="flare", linewidth=2.5)

        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='major', color='w', linewidth=1.0)
        ax.grid(b=True, which='minor', color='w', linewidth=0.5)
        ax.set(xlabel='Last Month Contacted', ylabel='Frequency of Success')

        plt.xticks(np.arange(3, 13, 1))
        locs, labels = plt.xticks()
        plt.xticks(locs, calendar.month_name[3::1])

        plt.axhline(y=success_freq.iloc[:, 0].mean(), color='b', linestyle='--', label="avg.")
        plt.axhline(y=success_freq.iloc[:, 1].mean(), color='g', linestyle='--', label="avg.")

        plt.legend()
        if plot == True:
            plt.show()

        # Plot number of contacts and number of success
        fig, ax = plt.subplots(2, 1)

        ax[0].bar(calendar.month_name[3::1], month_count[1], width=0.45, align="edge")
        ax[0].bar(calendar.month_name[3::1], month_count[0], width=0.45, align="edge")
        ax[0].bar(calendar.month_name[3::1], pos_outcomesPT[1], width=-0.45, align="edge")
        ax[0].bar(calendar.month_name[3::1], pos_outcomesPT[0], width=-0.45, align="edge")
        ax[0].set_ylabel('Number of People')
        plt.suptitle('Last Month Contacted')
        ax[0].legend(["Contacted (New)", "Contacted (Old)", "Success (New)", "Success (Old)"])

        ax[1].bar(calendar.month_name[3::1], month_count[1], width=0.45, align="edge")
        ax[1].bar(calendar.month_name[3::1], month_count[0], width=0.45, align="edge")
        ax[1].bar(calendar.month_name[3::1], pos_outcomesPT[1], width=-0.45, align="edge")
        ax[1].bar(calendar.month_name[3::1], pos_outcomesPT[0], width=-0.45, align="edge")

        plt.title("Zoom-In")
        plt.ylim(0, 800)

        if plot == True:
            plt.show()

        print("----------------Previous Campaign Info----------------")

        return None

    def missing_data_info(self, input_name, plot=False, agg_func="count"):
        print("\n----------------Missing Data Info----------------")
        if input_name not in self._feature_df.columns:
            print("Feature: ", input_name, " not in Feature Space.")
            print("----------------Missing Data Info----------------")
            return None

        print("Consider ONLY when ", input_name, " is missing.")
        current_data = pd.concat([self._feature_df.iloc[:, :-12], self._output_df], axis=1) # only consider inputs that can also be missing
        missing_df   = current_data[input_name].value_counts()

        print(missing_df)

        missing_df = current_data[(current_data[input_name] == "unknown") | (current_data[input_name] == 0.5)]
        if missing_df.shape[0] == 0:
            print("\nNo Missing Data!!")
            print("\n----------------Missing Data Info----------------")
            return None

        print("\n-->Random sample: ")
        print(missing_df.sample(20))

        fig, ax = plt.subplots(2, 4)

        i = 0
        feat_group1 = ["age", "job", "default", "houseLoan"]
        feat_group2 = ["marital", "education", "personalLoan", "y"]
        BIN = 1.0
        for feat in feat_group1:
            if len(missing_df[feat].unique()) > 2:
                bb = 2.0
                gg = sns.histplot(ax=ax[0, i], data=missing_df, x=feat, binwidth=bb, stat=agg_func)
            else:
                bb = 0.05
                gg = sns.histplot(ax=ax[0, i], data=missing_df, x=feat, binwidth=bb, binrange=[-BIN, BIN], stat=agg_func)

            if missing_df[(missing_df[feat] == "unknown")].shape[0] != 0:
                gg = sns.histplot(ax=ax[0, i],
                                  data=missing_df[(missing_df[feat] == "unknown")],
                                  x=feat, color=(1, 0, 0), binwidth=bb, stat=agg_func)
                gg.legend([None, "missing"])
            elif missing_df[(missing_df[feat] == 0.5)].shape[0] != 0:
                gg = sns.histplot(ax=ax[0, i],
                                  data=missing_df[(missing_df[feat] == 0.5)],
                                  x=feat, color=(1, 0, 0), binwidth=bb, legend=True, stat=agg_func)
                gg.legend([None, "missing"])
            if feat == "job":
                gg.set_xticklabels(gg.get_xticklabels(), rotation=-40)

            ax[0, i].get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[0, i].get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[0, i].grid(b=True, which='major', color='w', linewidth=1.0)
            ax[0, i].grid(b=True, which='minor', color='w', linewidth=0.5)

            i = i + 1
        i = 0
        for feat in feat_group2:
            if len(missing_df[feat].unique()) > 2:
                bb = 2.0
                gg = sns.histplot(ax=ax[1, i], data=missing_df, x=feat, binwidth=bb, stat=agg_func)
            else:
                bb = 0.05
                gg = sns.histplot(ax=ax[1, i], data=missing_df, x=feat, binwidth=bb, binrange=[-BIN, BIN], stat=agg_func)
            if missing_df[(missing_df[feat] == "unknown")].shape[0] != 0:
                gg = sns.histplot(ax=ax[1, i],
                                  data=missing_df[(missing_df[feat] == "unknown")],
                                  x=feat, color=(1, 0, 0), binwidth=bb, stat=agg_func)
                gg.legend([None, "missing"])
            elif missing_df[(missing_df[feat] == 0.5)].shape[0] != 0:
                gg = sns.histplot(ax=ax[1, i],
                                  data=missing_df[(missing_df[feat] == 0.5)],
                                  x=feat, color=(1, 0, 0), binwidth=bb, stat=agg_func)
                gg.legend([None, "missing"])
            if feat == "education":
                gg.set_xticklabels(gg.get_xticklabels(), rotation=-40)

            ax[1, i].get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[1, i].get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[1, i].grid(b=True, which='major', color='w', linewidth=1.0)
            ax[1, i].grid(b=True, which='minor', color='w', linewidth=0.5)

            i = i + 1

        plt.suptitle("Consider ONLY when {} is missing.".format(input_name))
        if plot:
            plt.show()

        print("----------------Missing Data Info----------------")
        return missing_df

    def del_examples(self, indices):
        print("\n--------Deleting Instances------------")
        print("Dropping indices: ", list(indices))
        print("Percent of data being deleted: ", (indices.shape[0] / self._feature_df.shape[0]) * 100)
        print("Feature matrix dim before deletion: ", self._feature_df.shape[0], ' x ', self._feature_df.shape[1])
        print("Size: ", indices.shape)

        current_data = pd.concat([self._feature_df, self._output_df, self._dura_df, self._current_model_per_df], axis=1)

        rnd_ind = np.random.choice(indices, 20)
        rnd_ind = list(rnd_ind)
        rnd_ind.sort()
        print("Sample of examples being dropped: ")
        print(current_data.loc[rnd_ind])

        current_data.drop(labels=indices, axis=0, inplace=True)

        self._output_df            = current_data["y"]
        self._dura_df              = current_data["duration"]
        self._current_model_per_df = current_data["ModelPrediction"]
        self._feature_df           = current_data.iloc[:, :-3]

        print("Feature matrix dim after deletion: ", self._feature_df.shape[0], ' x ', self._feature_df.shape[1])
        print("--------Deleting Instances------------")
        return current_data

    def intuitive_featSel(self, input_name):
        print("\n-------Deleting ENTIRE Feature----------")
        if input_name not in self._feature_df.columns:
            print(input_name, " not in feature matrix.")
            print("-------Deleting ENTIRE Feature----------")
            return None

        print("Dropping input: ", input_name)
        print(self._feature_df[input_name].value_counts())

        self._feature_df.drop(labels=input_name, axis=1, inplace=True)
        print("New feature matrix: ")
        print(self._feature_df.head(5))

        print("-------Deleting ENTIRE Feature----------")
        return None

    def scale(self):
        return None

    def numerical_encoding(self):
        # jobs - 1 hot
        # marital - 1 hot
        # education - int. label
        # commType - binary
        #       1 == "cellular"    0 == "telephone"
        # priorCampOutcome -
        # priorDay - cyclical
        # priorMonth - cyclical

    def merge_feature_values(self, input_name, value_list, replacement):
        # Merge values
        print("\n----------Merging----------")
        if input_name not in self._feature_df.columns:
            print("Feature: ", input_name, " not in Feature Space.")
            print("----------Merging----------")
            return None
        for val in value_list:
            if val not in self._feature_df[input_name].unique():
                print("Value: ", val, " not in Values of ", input_name)
                print("----------Merging----------")
                return None

        print("From: ", input_name, ". We are merging values: {", value_list, "} ")

        self._feature_df[input_name].replace(value_list, replacement, inplace=True)
        print(self._feature_df[input_name].value_counts())

        print("----------Merging----------")
        return None