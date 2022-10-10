from decimal import DivisionByZero
from typing import Type
import numpy as np

class Matrix:
    def __init__(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                raise TypeError(type(data))
            if not isinstance(data[0], list):
                raise TypeError(type(data))
            else:
                if not sum([len(row) == len(data[0]) for row in data]) == len(data):
                    raise AttributeError("matrix rows must be of the same length")
                self.data = data
                self.shape = (len(data), len(data[0]))
        elif isinstance(data, tuple):
            if not len(data) == 2:
                raise AttributeError("shape must be a tuple of 2 integers")
            if not isinstance(data[0], int) or not isinstance(data[1], int):
                raise TypeError(type(data))
            self.data = [[0 for _ in range(data[1])] for _ in range(data[0])]
        else:
            raise TypeError(type(data))
    
    def __add__(self, rhs):
        if not isinstance(rhs, Matrix):
            raise TypeError
        if not self.shape == rhs.shape:
            raise AttributeError("can only add matrices if same shape")
        new_data = [[a+b for a,b in zip(rowa, rowb)] for rowa, rowb in zip(self.data, rhs.data)]
        return (type(self)(new_data))
    
    def __radd__(self, rhs):
        return (rhs+self)

    def __sub__(self, rhs):
        if not isinstance(rhs, Matrix):
            raise TypeError(type(rhs))
        if not self.shape == rhs.shape:
            raise AttributeError("can only add matrices if same shape")
        new_data = [[a-b for a,b in zip(rowa, rowb)] for rowa, rowb in zip(self.data, rhs.data)]
        return (type(self)(new_data))

    def __rsub__(self, rhs):
        return (rhs - self)
    
    def __truediv__(self, rhs):
        if not isinstance(rhs, int) and not isinstance(rhs, float):
            raise TypeError(type(rhs))
        if rhs == 0:
            raise DivisionByZero
        new_data = [[x / rhs for x in row] for row in self.data]
        return (type(self)(new_data))
    
    def __rtruediv__(self, rhs):
        return self / (1 / rhs)

    def __mul__(self, rhs):
        if not isinstance(rhs, int) and not isinstance(rhs, float):
            raise TypeError(type(rhs))
        new_data = [[x * rhs for x in row] for row in self.data]
        return (type(self)(new_data))

    def __rmul__(self, rhs):
        return (self * rhs)

    def __str__(self):
        return(str(self.data))
    
    def __repr__(self):
        return (self.__class__.__name__+"("+str(self)+", shape="+str(self.shape)+")")
    
    def T(self):
        new_data = [[row[index] for row in self.data] for index in range(len(self.data[0]))]
        if len(new_data) == 0:
            new_data = [[]]
        return type(self)(new_data)

class Vector(Matrix):
    def __init__(self, data):
        Matrix.__init__(self, data)
        if not self.shape[0] == 1 and not self.shape[1] == 1:
            raise AttributeError("vector must be one dimensional")
    
    def dot(self, v):
        if not isinstance(v, Vector):
            return TypeError
        if not self.shape == v.shape:
            return AttributeError("dot product can only be made between vectors of the same shape")
