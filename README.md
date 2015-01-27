FastUtils
=============================
A naive version of dot product based on cython. It's multi-threading.

Why FastUtils
---------------------
Python is a good programming language but is too slow. 
**GIL**(Global Interpreter Lock) bothers us too much when we want to use more CPUs in our codes.
So I write a tool with cython(a optimising static compiler which can be used to write C extensions for python).
It's easy to make the job run parallelly with `cython.parallel`.
**BLAS**(Basic Linear Algebra Subprograms) is also used in the code too.

It's no doubt that python has already had a powerful lib *numpy* and *scipy* to do scientific caculations.
`numpy.dot` is also support multi-threading if you compiled it with optimized BLAS library like *ATLAS* or *OpenBLAS*.
The code is only written because I want to practise how to write C extensions for python.

About the Code
---------------------
The code is in `fast_utils.pyx`. There are two functions: `fast_dot()` and `fast_dot_blas()`. 
The latter one uses blas to do calculations faster. We should remember that 
the two input matrix *A* and *B* shouldn't have been `numpy.transpose()` if we want to get the answer
right. If we want to do `A * B` we can call `fast_dot(A,B,Result,n_jobs=2)` or 
`fast_dot_blas(A,B_Transpose,n_jobs=2)`. 
The input matrix should be like this `numpy.array([[1,2],[2,4]], dtype=float32)`.
Run `test_utils.py` will make you more clear. 

Speed Test
----------------------
* Jobs: dot product between `A(2000,3000)` and `B(3000,2000)`
* Env: ubuntu 14, x64, Intel(R) Core(TM) i7 CPU, 2.93GHz
<table class="table table-bordered table-striped table-condensed">
<tr>
	<th>method</th>
	<th>core(s)</th>
	<th>time</th>
</tr><tr>
	<td>np.dot</td>
	<td>1</td>
	<td>10.85s</td>
</tr><tr>
	<td>fast_dot</td>
	<td>1</td>
	<td>60.97s</td>
</tr><tr>
	<td>fast_dot_blas</td>
	<td>1</td>
	<td>10.74s</td>
</tr><tr>
	<td>fast_dot</td>
	<td>8</td>
	<td>22.50s</td>
</tr><tr>
	<td>fast_dot_blas</td>
	<td>8</td>
	<td>2.06s</td>
</tr>
* Here the `np.dot` is compiled with the normal blas version(not optimized), so the speed is not very fast. `fast_dot` is pure C and its 
speed is too slow compared to the other two methods.

Quick Start
----------------------
* Some dependencies: install python, cython(0.20.2 or higher), numpy, scipy 
* Compiling: `python setup build`
* Copy the lib to your work space: `cp build/lib.linux-x86_64-2.7/fast_utils.so ./your_workspace` 
* usage: `from fast_utils import fast_dot` or `from fast_utils import fast_dot_blas`

More infomation
---------------------
If you have any questions or suggestions, feel free to contact me(shaoyf2011@gmail.com).
