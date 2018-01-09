from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from fancyimpute import SoftImpute

# TODO correctly arrange train and test data
# TODO one hot encode data
data_csv = pd.read_csv("py_true_train.csv", delimiter=",")
data_csv_test = pd.read_csv("py_true_test.csv", delimiter=",")
# char_cols = data_csv.dtypes.pipe(lambda x: x[x == 'object']).index
# for c in char_cols:
#     data_csv[c] = pd.factorize(data_csv[c])[0]
#     data_csv_test[c] = pd.factorize(data_csv_test[c])[0]

numColumns = len(data_csv.columns)
dataset = data_csv.values
explanatory = dataset[:, 0:numColumns - 1]
response = dataset[:, numColumns - 1]
print(response)


