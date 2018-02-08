from pandas import read_csv
from sklearn.decomposition import PCA
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, datasets

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# print(X)

sc = StandardScaler()
X_std = sc.fit_transform(X)
# print(X_std)

pca = decomposition.PCA(n_components=3)
X_std_pca = pca.fit_transform(X_std)
print(X_std_pca.shape)
print(X_std_pca)
df = pandas.DataFrame(X_std_pca)
# df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# pyplot.show()
X_rec = pca.inverse_transform(X_std_pca)
print(X_rec)