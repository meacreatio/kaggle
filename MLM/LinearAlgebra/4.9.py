from numpy import array
from numpy import vstack
from numpy import hstack
from numpy import fromstring
a1 = array([1,2,3])
print(a1)
a2 = array((4,5,6))
print(a2)
a3 = vstack((a1, a2))
print(a3)
print(a3.shape)

#4.11
a4 = hstack((a1, a2))
print(a4)
print(a4.shape)

str = "1,2,3"
a5 = fromstring(str, sep=",")
print(a5)
