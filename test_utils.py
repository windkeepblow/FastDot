import time
import numpy as np
from numpy import float32 as REAL
from fast_utils import fast_dot
from fast_utils import fast_dot_blas

def main():
    A = np.array(np.random.random((2000,3000)),dtype=REAL) 
    B = np.array(np.random.random((3000,2000)),dtype=REAL)
    C = np.array(np.random.random((2000,3000)),dtype=REAL) 
    print "A.shape", A.shape
    print "B.shape", B.shape
    ''' 
    A = np.array([[1.2,2.3,3.4,5.5],[31,4,3,5],[2,3,67,45]], dtype=REAL)
    B = np.array([[5.9,6,3],[7,8,3],[1,2,3],[11,34,55.5]], dtype=REAL)
    C = np.array([[5.9,7,1,11],[6,8,2,34],[3,3,3,55.5]], dtype=REAL)
    ''' 
    st = time.time()
    result_1 = np.empty((A.shape[0], B.shape[1]), dtype=REAL)
    fast_dot(A, B, result_1, 1)
    ed = time.time()
    print "fast_dot time:%fs"%(ed-st)
    
    st = time.time()
    result_2 = np.dot(A,B)
    ed = time.time()
    print "np.dot time:%fs"%(ed-st)
    
    st = time.time()
    result_3 = np.empty((A.shape[0], C.shape[0]), dtype=REAL)
    fast_dot_blas(A, C, result_3, 1)
    ed = time.time()
    print "fast_dot_blas time:%fs"%(ed-st)
    '''
    print result_1
    print result_2
    print result_3
    '''
if __name__=="__main__":
    main()
