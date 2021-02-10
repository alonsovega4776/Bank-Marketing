"""
Alonso Vega
January 22, 2021
Driver: From here we can call the other classes to obtain our classifier.
"""

import DataWrangler
import pandas as pd

# Initial Cleaning
bankingData = DataWrangler.DataWrangler("DSADataSet.csv")

# Prior campaign info
bankingData.prior_campaign(plot=False)

# Missing Data
# Loan data
df_missing_1 = bankingData.missing_data_info("personalLoan", False)
df_missing_2 = bankingData.missing_data_info("houseLoan", False)
bankingData.del_examples(df_missing_2.index)
df_missing_2 = bankingData.missing_data_info("personalLoan", plot=False)

# default missing data
bankingData.missing_data_info(input_name="default", plot=False)
# job missing data
bankingData.missing_data_info(input_name="job", plot=False)
#eduaction
bankingData.missing_data_info(input_name="education", plot=False)

merging = ["unknown", "illiterate"]
bankingData.merge_feature_values(input_name="education", value_list=merging, replacement="less.4y" )
bankingData.merge_feature_values(input_name="education", value_list=merging, replacement="less.4y" )

# job missing data
bankingData.missing_data_info(input_name="job", plot=False)

# merging unemployed and unknown
merging[1] = "unemployed"
bankingData.merge_feature_values(input_name="job", value_list=merging, replacement="unemployed")
bankingData.missing_data_info(input_name="job", plot=True)

# marital missing data
bankingData.missing_data_info(input_name="marital", plot=False)

# just consider marital to be either married or NOT married
merging[1] = "divorced"
merging.append("single")
bankingData.merge_feature_values("marital", merging, "not.married")

# for default merge 1.0 and 0.5
# there is only 2 1.0
# consider feature default as {NO, MAYBE}
bankingData.merge_feature_values("default", [1.0, 0.5], 1)
bankingData.merge_feature_values("default", [1.0, 0.5], 1)
bankingData.missing_data_info("default", plot=True)

# check for missing data
for feat in df_missing_1.columns:
    bankingData.missing_data_info(feat, True)

# Delete priorCampDays
# its redundent
bankingData.numerical_encoding("priorCampOutcome", ["nonexistent", "failure", "success"])  # only do this to get heatmap
#bankingData.heat_map(otro=["priorCampDays", "priorCampOutcome", "priorCampContacts"])
bankingData.intuitive_featSel("priorCampDays")

# Outliers
bankingData.remove_OL("age", plot=False, DELETE=True)

# Getting rid of object types in our Data
# integer
replace_list_ordered = ["less.4y", "basic.4y", "basic.6y", "basic.9y",
                        "high.school", "professional.course", "university.degree"]
bankingData.numerical_encoding("education", unq_values_ordered=replace_list_ordered)

# binary
bankingData.numerical_encoding("marital", method="binary")
bankingData.numerical_encoding("commType", method="binary")

replace_list_ordered = list(bankingData.get_features()["job"].unique())
bankingData.numerical_encoding("job", unq_values_ordered=replace_list_ordered)


'''
# hot vectors
bankingData.numerical_encoding("job", method="k dum")
bankingData.numerical_encoding("priorCampOutcome", method="k-1 dum")
'''

# Fix Scaling
bankingData.scale("consPriceIdx", method="minmax")
bankingData.scale("empVariation", method="std")
bankingData.scale("euribor3Mon", method="minmax")
bankingData.scale("employees", method="minmax")
bankingData.scale("consConfidenceIdx", method="minmax", minmax_range=(-1, 1))

'''
bankingData.scale("age", method="minmax")
bankingData.scale("education", "minmax")
bankingData.scale("priorDay", "minmax")
bankingData.scale("priorMonth", "minmax")
bankingData.scale("currentCampContacts", "minmax")
'''

# Descritize
bankingData.discretize("employees")
bankingData.discretize("euribor3Mon")
bankingData.discretize("consConfidenceIdx")
bankingData.discretize("consPriceIdx")
bankingData.discretize("empVariation")


# Get clean Data
X_mat, y_vect = bankingData.get_clean_data()
print(X_mat.head(5))

# make csv of Data
bankingData.make_csv()
