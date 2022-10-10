from matrix import Matrix, Vector

# m = Matrix([])
# m = Matrix([[2, 2],[1, 4, 5]])
# m = Matrix(None)
m = Matrix((3,4))
m1 = Matrix([[2,2,5],[-5.6,7,8]])
m2 = Matrix([[1,1,1],[1,1,1]])
print(m1+m2)
print(m1-m2)
print(m1/4)
print(4/m1)
print(m1 * 4)
print(2 * m1)
print(repr(m1))
# print(m1/0)
print(m1.T())
m0 = Matrix([[]])
print(m0.T())
# v = Vector(None)
# v = Vector([[2,2,5],[-5.6,7,8]])
v = Vector([[1,2,3,4]])
print(repr(v))
v1 = Vector([[1, 7, 8]])
v2 = Vector([[2, 9 , 1]])
print(repr(v1+v2))
print(v1.dot(v2))

