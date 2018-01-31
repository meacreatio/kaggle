from numpy import array
data = array([11, 22, 33, 44, 55])
print(data.shape)

data = array([[11, 22], [44, 55], [77, 88]])
data = array(data)
print(data.shape) # 3 rows 2 columns

# access the touple
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])

data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape from 1D to 2D
data = data.reshape((data.shape[0], 1)) # convert to 5 rows 1 columns
print(data.shape)

# 2D to 3D
data = [[11, 22],
[33, 44],
[55, 66]]
# array of data
data = array(data)
print(data.shape)
# reshape from 2D to 3D
data = data.reshape((data.shape[0], data.shape[1], 1))
# rows (samples), columns (time steps), features 3, 2, 1
print(data.shape)
