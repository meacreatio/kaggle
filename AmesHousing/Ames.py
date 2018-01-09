from numpy import loadtxt
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_csv = read_csv("train.csv", delimiter=",", header=None)
numColumns = len(data_csv.columns)
dataset = data_csv.values
explanatory = dataset[:, 0:numColumns - 1]
response = dataset[:, numColumns - 1]
print(explanatory)

# split into explanatory and response
