from __future__ import print_function
from builtins import range
import numpy as np
import ipdb
import os.path


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


def clean_vec(f, name, *args):
    f.write("c_free(")
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    # else:
        # f.write("c_float * ")
    f.write("%s);\n" % name)


def write_mat_sparse(f, A, name, *args):
    m = A.shape[0]
    n = A.shape[1]

    f.write("// Matrix " + name + "\n")
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


def clean_mat(f, name, *args):

    # Clean data vector
    f.write("c_free(")
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    f.write("%s->x);\n" % name)

    # Clean index vector
    f.write("c_free(")
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    f.write("%s->i);\n" % name)

    # Clean col pointer vector
    f.write("c_free(")
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    f.write("%s->p);\n" % name)

    # Clean matrix
    f.write("c_free(")
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    f.write("%s);\n" % name)


def generate_code(P, q, A, lA, uA, problem_name):

    # Get problem dimension
    n = P.shape[0]
    m = A.shape[0]

    # GENERATE HEADER FILE
    #
    #
    f = open("qptests/" + problem_name + "/" + problem_name + ".h", "w+")

    # Add definition check
    f.write("#ifndef " + problem_name.upper() + "_H\n")
    f.write("#define " + problem_name.upper() + "_H\n")

    # Add Includes
    f.write("#include \"osqp.h\"\n")
    f.write("\n\n")

    # Add function generation
    f.write("/* function to generate problem data structure */\n")
    f.write("Data * generate_problem_%s();\n\n" % problem_name)

    # Add function cleanup
    f.write("/* function to clean problem data structure */\n")
    f.write("c_int clean_problem_%s(Data * data);\n\n" % problem_name)

    f.write("#endif\n")

    f.close()

    # GENERATE .C FILE
    #
    #
    f = open("qptests/" + problem_name + "/" + problem_name + ".c", "w+")
    f.write("#include \"" + problem_name + ".h\"\n")
    f.write("\n")

    # Add function GENERATION
    f.write("/* function to generate problem data structure */\n")
    f.write("Data * generate_problem_%s(){\n\n" % problem_name)

    # Initialize structure data
    f.write("Data * data = (Data *)c_malloc(sizeof(Data));\n\n")

    # Write problem dimensions
    f.write("// Problem dimensions\n")
    write_int(f, n, "n", "data")
    write_int(f, m, "m", "data")
    f.write("\n")

    # Write problem vectors
    f.write("// Problem vectors\n")
    write_vec_float(f, lA, "lA", "data")
    write_vec_float(f, uA, "uA", "data")
    write_vec_float(f, q, "q", "data")
    f.write("\n")

    # Write matrix A
    write_mat_sparse(f, A, "A", "data")
    write_mat_sparse(f, P, "P", "data")

    # Return data and end function
    f.write("return data;\n\n")
    f.write("}\n\n")

    # Add function CLEANUP
    f.write("/* function to clean problem data structure */\n")
    f.write("c_int clean_problem_%s(Data * data){\n\n" % problem_name)

    # Free vectors
    f.write("// Clean vectors\n")
    clean_vec(f, "lA", "data")
    clean_vec(f, "uA", "data")
    clean_vec(f, "q", "data")
    f.write("\n")

    # Free matrices
    f.write("//Clean Matrices\n")
    clean_mat(f, "A", "data")
    clean_mat(f, "P", "data")
    f.write("\n")

    # Clean data structure
    f.write("c_free(data);\n")

    f.write("return 0;\n\n")
    f.write("}\n\n")

    # GENERATE TEST FILE
    #
    #
    if os.path.isfile("qptests/" + problem_name + "/test_" +
                      problem_name + ".h"):
        print("Test file for %s_test already present. Skip creating it\n" \
            % problem_name)
    else:
        f = open("qptests/" + problem_name + "/test_" +
                 problem_name + ".h", "w+")
        f.write("#include \"osqp.h\"\n")
        f.write("#include \"minunit.h\"\n")
        f.write("#include \"qptests/" + problem_name +
                "/" + problem_name + ".h\"\n")

        # Add definition check
        f.write("#ifndef TEST_" + problem_name.upper() + "_H\n")
        f.write("#define TEST_" + problem_name.upper() + "_H\n\n")

        f.write("static char * test_" + problem_name + "()\n{\n")
        f.write("/* local variables */\n")
        f.write("c_int exitflag = 0;  // No errors\n")
        f.write("\n")
        f.write("// Problem settings\n")
        f.write("Settings * settings = ")
        f.write("(Settings *)c_malloc(sizeof(Settings));\n")
        f.write("\n")

        f.write("// Structures\n")
        f.write("Work * work;  // Workspace\n\n")

        f.write("// Generate problem data\n")
        f.write("Data * data = generate_problem_diesel();\n\n")

        f.write("c_print(\"\\nTest %s\\n\");\n" % problem_name)
        f.write("c_print(\"-------------\\n\");\n\n")

        f.write("// Define Solver settings as default\n")
        f.write("set_default_settings(settings);\n\n")

        f.write("// Setup workspace\n")
        f.write("work = osqp_setup(data, settings);\n\n")

        f.write("if (!work) {\n")
        f.write("c_print(\"Setup error!\\n\");\n")
        f.write("exitflag = 1;\n")
        f.write("}\n")
        f.write("else {\n")
        f.write("// Solve Problem\n")
        f.write("osqp_solve(work);\n\n")
        f.write("// Clean workspace\n")
        f.write("osqp_cleanup(work);\n\n")
        f.write("}\n\n")

        f.write("// Cleanup data\n")
        f.write("clean_problem_diesel(data);\n\n")
        f.write("mu_assert(\"\\nError in %s test.\", exitflag == 0 );\n\n"
                % problem_name)

        f.write("//Cleanup\n")
        f.write("c_free(settings);\n\n")

        f.write("return 0;\n")
        f.write("}\n\n")

        f.write("#endif\n")

        f.close()
