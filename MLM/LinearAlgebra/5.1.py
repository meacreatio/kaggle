from numpy import array
data = [11, 22, 33, 44, 55]
data = array(data)
print(data)
print(type(data))

data = [[11, 22], [33, 44], [55, 66]]
data = array(data)
print(data)
print(data[0,])
print(data[-1,])
print(data[-2,])
print(data[1, 0])

print(data[0 : 1, 0 : 1])

data = array([11, 22, 33, 44, 55])
print(data[-2:])