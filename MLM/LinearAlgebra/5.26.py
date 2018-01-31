from numpy import array
data = array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
# slice all rows and columns except for the last one, slice all rows from the last column
X, y = data[:, : -1], data[:, -1]
print(X)
print(y)

split = 2
train, test = data[:split, :], data[split:, :]
print("")
print(train)
print("")
print(test)