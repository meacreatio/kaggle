from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
skew = data.skew()
print(skew)
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# scatter_matrix(data)
# data.hist()
pyplot.show()

