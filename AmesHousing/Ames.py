from numpy import loadtxt
import pandas
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pandas.read_csv("train.csv", delimiter=",")
print(dataset)