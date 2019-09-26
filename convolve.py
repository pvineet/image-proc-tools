
import numpy as np
import time
import random
from multiprocessing import Process, cpu_count
from scipy.signal import convolve2d # to generate reference output

# List of lists to store a matrix is slower than using numpy.
# Using numpy to store the matrix value.
# Numpy or Scipy are not used to perform convolution
# Given filter K=[-1, 0, 1]
# Let us flip the filter rather that flipping the image

'''
    This code calculated valid convolution output
    Valid mode in scipy.signal.convolve -
        The output consists only of those elements that do not rely on the zero-padding. In ‘valid’ mode, either in1 or in2 must be at least as large as the other in every dimension.
'''

class Solution:
    def __init__(self, n_row, n_col, int_width=8):
        self.n_row = n_row
        self.n_col = n_col
        self.int_width = int_width
        self.max_int = 2**self.int_width-1
        self.k  = [1, 0, -1]
        self.test_kernel_v = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
        self.test_kernel_h = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        self.matrix = np.pad(np.random.randint(0, self.max_int,(self.n_row, self.n_col)),(2,2),mode='constant',constant_values=0)

    def get_num_cpu_cores(self):
        return cpu_count()

    def get_np_dtype(self):
        '''
            Right now the np dtype and int_width are decoupled
        '''
        pass

    def  conv_hor(self):
        '''
            function to compute horizontal computation
        '''
        result = np.empty((self.n_row+2, self.n_col+2), dtype=np.uint8)
        for i in range(self.n_row+2):
            for j in range(self.n_col+2):
                result[i,j] = self.matrix[i,j] - self.matrix[i,j+2]
        return result

    def conv_hor_using_transpose(self):
        '''
            Transpose out input matrix and compute the vertical conv
        '''
        result = list()
        _matrix = np.transpose(self.matrix)
        for i in range(self.n_col):
            result.append(_matrix[i,] - _matrix[i+2,])
        return np.array(result)

    def  conv_ver(self):
        '''
            function to compute verticaal computation
        '''
        result = list()
        for i in range(self.n_row):
            result.append(self.matrix[i,] - self.matrix[i+2,])
        return np.array(result)

    def test_conv_hor(self, in_matrix):
        '''
            in_matrix is the output of the methods implemented above
        '''
        ref_output = convolve2d(self.matrix, self.test_kernel_h)
        if not ref_output.shape == in_matrix.shape:
            print("Input Matrix and Reference Matrix shapes differ {} {}".format(ref_output.shape, in_matrix.shape))
            print(ref_output)
            print(in_matrix)
            return False
        if (in_matrix == ref_output).all():
            print("Input matrix is correct")
            return True
        else:
            print("Input matrix is wrong")
            return False

    def test_conv_ver(self, in_matrix):
        '''
            in_matrix is the output of the methods implemented above
        '''
        ref_output = convolve2d(self.matrix, self.test_kernel_v)
        if not ref_output.shape == in_matrix.shape:
            print("Input Matrix and Reference Matrix shapes differ")
            return False
        if (in_matrix == ref_output).all():
            print("Input matrix is correct")
            return True
        else:
            print("Input matrix is wrong")
            return False

s = Solution(4,4)
# s.test_conv_hor(s.conv_hor())
print(s.matrix)
print("==============================")
print(s.conv_hor_using_transpose())
print("==============================")
print(s.conv_ver())
print("==============================")
print(convolve2d(s.matrix,s.test_kernel_h))
print("==============================")
print(convolve2d(s.matrix,s.test_kernel_v))
