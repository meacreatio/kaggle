from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from kaggle.Utils.UtilsDataFrame import delete_column
from kaggle.Utils.UtilsDataFrame import print_na_count
from kaggle.Utils.UtilsDataFrame import remove_header

from fancyimpute import MICE


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

# factorize categorical values
char_cols = df_train.dtypes.pipe(lambda x: x[x == 'object']).index
for c in char_cols:
    df_train[c] = pd.factorize(df_train[c])[0]
    # data_csv_test[c] = pd.factorize(data_csv_test[c])[0]

# TODO impute missing values
df_train.to_csv("before.csv")
df_train = MICE().complete(df_train)
df_train = pd.DataFrame(df_train)
df_train.to_csv("temp.csv")

numColumns = len(df_train.columns)
dataset = df_train.values

explanatory = dataset[:, 0:numColumns - 1]
response = dataset[:, numColumns - 1]
print(explanatory)


