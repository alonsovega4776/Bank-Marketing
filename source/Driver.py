"""
Alonso Vega
January 22, 2021
Driver: From here we can call the other classes to obtain our classifier.
"""
import DataWrangler

bankingData = DataWrangler.DataWrangler("DSADataSet.csv")
bankingData.unique_observation("job", plot=True)
bankingData.missing_data_stats("job")