from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
skew = data.skew()
print(skew)
# data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# scatter_matrix(data)
# data.hist()

# pyplot.show()
array = data.values
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(1, 2))
rescaledX = scaler.fit_transform(X)
# scaler = StandardScaler().fit(X)
# rescaledX = scaler.transform(X)
# scaler = Normalizer().fit(X)
# normalizedX = scaler.transform(X)

df_print = pandas.DataFrame(rescaledX)
# df_print = df_print.apply(numpy.log)
# print(df_print)
# df_print.hist()

print(list(df_print.columns.values))
df_print[0] = numpy.log(df_print[0])
df_print.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()