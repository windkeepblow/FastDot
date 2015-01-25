cimport numpy as np
from libc.stdio cimport printf
from cython.parallel import prange

ctypedef np.float32_t REAL_t
"""use blas"""
from scipy.linalg.blas import fblas
cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)

"""
R = A * B (warning: A, B should't have been numpy.array.transposed!)
@R: result matrix
@n_jobs: num of CPUs to be used
"""
def fast_dot(A, B, R, n_jobs):
    '''prepare data'''
    cdef int A_row = <int>A.shape[0]
    cdef int A_col = <int>A.shape[1]
    cdef int B_row = <int>B.shape[0]
    cdef int B_col = <int>B.shape[1]
    cdef REAL_t *mat_1 = <REAL_t *>np.PyArray_DATA(A) 
    cdef REAL_t *mat_2 = <REAL_t *>np.PyArray_DATA(B) 
    cdef REAL_t *result = <REAL_t *>np.PyArray_DATA(R)    
    cdef int cores = <int>n_jobs
    '''begin dot''' 
    cdef int i=0, j=0, k=0
    for i in prange(A_row, nogil=True, num_threads=cores):
        for j in xrange(B_col):
            result[i*B_col+j] = <REAL_t>0.0
            for k in xrange(A_col):
                result[i*B_col+j] += mat_1[i*A_col+k] * mat_2[k*B_col+j] 

"""
R = A * B^T (warning: A, B should't have been numpy.array.transposed!)
@R: result matrix
@n_jobs: num of CPUs to be used
"""
cdef int ONE = 1
def fast_dot_blas(A, B, R, n_jobs):
    '''prepare data'''
    cdef int A_row = <int>A.shape[0]
    cdef int A_col = <int>A.shape[1]
    cdef int B_row = <int>B.shape[0]
    cdef int B_col = <int>B.shape[1]
    cdef REAL_t *mat_1 = <REAL_t *>np.PyArray_DATA(A) 
    cdef REAL_t *mat_2 = <REAL_t *>np.PyArray_DATA(B) 
    cdef REAL_t *result = <REAL_t *>np.PyArray_DATA(R)    
    cdef int cores = <int>n_jobs
    '''check whether fblas.sdot return float or double'''
    cdef int use_version = <int>0
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = <int>1
    cdef double d_res
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    if (abs(d_res - expected) < 0.0001):
        use_version = <int>1
    '''begin dot''' 
    cdef int i=0, j=0,k=0
    for i in prange(A_row, nogil=True, num_threads=cores):
        for j in xrange(B_row):
            if use_version:
                result[i*B_row+j] = <REAL_t>dsdot(&A_col, &mat_1[i*A_col], &ONE, &mat_2[j*B_col], &ONE)
            else:
                result[i*B_row+j] = <REAL_t>sdot(&A_col, &mat_1[i*A_col], &ONE, &mat_2[j*B_col], &ONE)

                
