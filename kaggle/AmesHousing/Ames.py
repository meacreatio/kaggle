from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from kaggle.Utils.UtilsDataFrame import delete_column
from kaggle.Utils.UtilsDataFrame import print_na_count
from kaggle.Utils.UtilsDataFrame import remove_header
# from fancyimpute import SoftImpute


df_train = pd.read_csv("train.csv", delimiter=",")
delete_column(df_train, 'Id')

# delete columns with large amounts of missing data
delete_column(df_train, 'Alley')
delete_column(df_train, 'FireplaceQu')
delete_column(df_train, 'PoolQC')
delete_column(df_train, 'Fence')
delete_column(df_train, 'MiscFeature')

# remove headers
remove_header(df_train)

# TODO one hot encode data

# TODO correctly arrange train and test data

data_csv_test = pd.read_csv("test.csv", delimiter=",")
# char_cols = data_csv.dtypes.pipe(lambda x: x[x == 'object']).index
# for c in char_cols:
#     data_csv[c] = pd.factorize(data_csv[c])[0]
#     data_csv_test[c] = pd.factorize(data_csv_test[c])[0]

numColumns = len(df_train.columns)
dataset = df_train.values

explanatory = dataset[:, 0:numColumns - 1]
response = dataset[:, numColumns - 1]
print(explanatory)


