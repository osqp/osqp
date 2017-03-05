"""
Utilities to generate embedded C code from OSQP sources
"""
# Compatibility with Python 2
from __future__ import print_function
# from builtins import dict
# from builtins import range

# import scipy.sparse as spa
import numpy as np
# import os.path

def write_int(f, x, name, *args):
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s = %i;\n" % (name, x))
    else:
        f.write("c_int %s = %i;\n" % (name, x))


def write_float(f, x, name, *args):
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s = %.20f;\n" % (name, x))
    else:
        f.write("c_float %s = %.20f;\n" % (name, x))


def write_vec_int(f, x, name, *args):
    n = len(x)
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    else:
        f.write("c_int * ")
    f.write("%s = c_malloc(%i * sizeof(c_int));\n" % (name, n))

    for i in range(n):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s[%i] = " % (name, i))
        f.write("%i;\n" % x[i])


def write_vec_float(f, x, name, *args):
    n = len(x)
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    else:
        f.write("c_float * ")
    f.write("%s = c_malloc(%i * sizeof(c_float));\n" % (name, n))

    for i in range(n):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s[%i] = " % (name, i))
        if x[i] == np.inf:
            f.write("OSQP_INFTY;\n")
        elif x[i] == -np.inf:
            f.write("-OSQP_INFTY;\n")
        else:
            f.write("%.20f;\n" % x[i])


def write_mat_sparse(f, A, name, *args):
    m = A.shape[0]
    n = A.shape[1]

    f.write("\n// Matrix " + name + "\n")
    f.write("//")
    f.write("-"*(len("Matrix  ") + len(name)) + "\n")

    # Allocate Matrix
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    else:
        f.write("csc * ")
    f.write(name + " = c_malloc(sizeof(csc));\n")

    # Write dimensions and number of nonzeros
    if any(args):
        write_int(f, m, "m", args, name)
        write_int(f, n, "n", args, name)
        write_int(f, -1, "nz", args, name)
        write_int(f, A.nnz, "nzmax", args, name)
    else:
        write_int(f, m, "m", name)
        write_int(f, n, "n", name)
        write_int(f, -1, "nz", name)
        write_int(f, A.nnz, "nzmax", name)

    for arg in args:
        f.write("%s->" % arg)
    f.write("%s->" % name)
    f.write("x = c_malloc(%i * sizeof(c_float));\n" % A.nnz)
    for i in range(A.nnz):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s->" % name)
        f.write("x[%i] = %.20f;\n" % (i, A.data[i]))

    for arg in args:
        f.write("%s->" % arg)
    f.write("%s->" % name)
    f.write("i = c_malloc(%i * sizeof(c_int));\n" % A.nnz)
    for i in range(A.nnz):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s->" % name)
        f.write("i[%i] = %i;\n" % (i, A.indices[i]))

    for arg in args:
        f.write("%s->" % arg)
    f.write("%s->" % name)
    f.write("p = c_malloc((%i + 1) * sizeof(c_int));\n" % n)
    for i in range(A.shape[1] + 1):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s->" % name)
        f.write("p[%i] = %i;\n" % (i, A.indptr[i]))

    # Do the same for i and p
    f.write("\n")
