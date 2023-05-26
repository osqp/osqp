from scipy import sparse
import numpy as np


def write_int(f, x, name, *args):
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s = %i;\n" % (name, x))
    else:
        f.write("OSQPInt %s = %i;\n" % (name, x))


def write_float(f, x, name, *args):
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s = %.20f;\n" % (name, x))
    else:
        f.write("OSQPFloat %s = %.20f;\n" % (name, x))


def write_vec_int(f, x, name, *args):
    n = len(x)
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    else:
        f.write("OSQPInt* ")
    f.write("%s = (OSQPInt*) c_malloc(%i * sizeof(OSQPInt));\n" % (name, n))

    for i in range(n):
        for arg in args:
            f.write("%s->" % arg)
        f.write("%s[%i] = " % (name, i))
        f.write("%i;\n" % x[i])

    f.write("\n")

def write_vec_float(f, x, name, *args):
    n = len(x)
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    else:
        f.write("OSQPFloat* ")
    f.write("%s = (OSQPFloat*) c_malloc(%i * sizeof(OSQPFloat));\n" % (name, n))

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

    f.write("\n")

def clean_vec(f, name, *args):
    f.write("c_free(")
    if any(args):
        for arg in args:
            f.write("%s->" % arg)
    # else:
        # f.write("OSQPFloat * ")
    f.write("%s);\n" % name)


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
        f.write("OSQPCscMatrix* ")
    f.write(name + " = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));\n")

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
    if min(m,n) == 0 or A.nnz == 0:
        f.write("%s->x = OSQP_NULL;\n" % name)
    else:
        f.write("%s->" % name)
        f.write("x = (OSQPFloat*) c_malloc(%i * sizeof(OSQPFloat));\n" % A.nnz)
        for i in range(A.nnz):
            for arg in args:
                f.write("%s->" % arg)
            f.write("%s->" % name)
            f.write("x[%i] = %.20f;\n" % (i, A.data[i]))

    for arg in args:
        f.write("%s->" % arg)
    if min(m,n) == 0 or A.nnz == 0:
        f.write("%s->i = OSQP_NULL;\n" % name)
    else:
        f.write("%s->" % name)
        f.write("i = (OSQPInt*) c_malloc(%i * sizeof(OSQPInt));\n" % A.nnz)
        for i in range(A.nnz):
            for arg in args:
                f.write("%s->" % arg)
            f.write("%s->" % name)
            f.write("i[%i] = %i;\n" % (i, A.indices[i]))

    for arg in args:
        f.write("%s->" % arg)
    f.write("%s->" % name)
    f.write("p = (OSQPInt*) c_malloc((%i + 1) * sizeof(OSQPInt));\n" % n)
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


def generate_problem_data(P, q, A, l, u, problem_name, sols_data={}):
    """
    Generate test problem data.

    The additional structure sols_data defines the additional vectors/scalars
    we need to perform the tests
    """
    # Get problem dimension
    n = P.shape[0]
    m = A.shape[0]

    #
    # GENERATE HEADER FILE
    #
    f = open(problem_name + "/" + problem_name + "_" + "data.h", "w")

    # Add definition check
    f.write("#ifndef " + problem_name.upper() + "_DATA_H\n")
    f.write("#define " + problem_name.upper() + "_DATA_H\n")

    # Add Includes
    f.write("#include \"osqp_api.h\"\n")
    f.write("#include \"osqp_tester.h\"\n")
    f.write("\n\n")

    f.write("/* Test case's QP problem data */\n")
    f.write("class %s_prob_data : public OSQPTestData {\n" % problem_name)
    f.write("public:\n")
    f.write("    %s_prob_data();\n" % problem_name)
    f.write("    ~%s_prob_data() = default;\n" % problem_name)
    f.write("};\n\n")

    #
    # Create additional data structure
    #
    f.write("/* Test case's additional data and solution */\n")
    f.write("class %s_sols_data {\n" % problem_name)
    f.write("public:\n")
    f.write("    %s_sols_data();\n" % problem_name)
    f.write("    ~%s_sols_data();\n\n" % problem_name)
    # Generate further data and solutions
    for key, value in sols_data.items():
        if isinstance(value, str):
            # Status test get from C code
            f.write("    OSQPInt %s;\n" % key)
        # Check if it is an array or a scalar
        elif isinstance(value, np.ndarray):
            if isinstance(value.flatten(order='F')[0], int):
                f.write("    OSQPInt* %s;\n" % key)
            elif isinstance(value.flatten(order='F')[0], float):
                f.write("    OSQPFloat* %s;\n" % key)
        else:
            if isinstance(value, int):
                f.write("    OSQPInt %s;\n" % key)
            elif isinstance(value, float):
                f.write("    OSQPFloat %s;\n" % key)
    f.write("};\n\n")

    #
    # Creator for the QP test data and additional test case data
    #
    f.write("/* Create test case data */\n")
    f.write("class %s_test_fixture : public OSQPTestFixture {\n" % problem_name)
    f.write("public:\n")
    f.write("    %s_test_fixture() : OSQPTestFixture()\n" % problem_name)
    f.write("        {\n")
    f.write("            data.reset(new %s_prob_data());\n" % problem_name)
    f.write("            sols_data.reset(new %s_sols_data());\n" % problem_name)
    f.write("        }\n")
    f.write("    ~%s_test_fixture() = default;\n\n" % problem_name)
    f.write("protected:\n")
    f.write("    std::unique_ptr<%s_sols_data> sols_data;\n" % problem_name)
    f.write("};\n\n")

    # Close header file
    f.write("#endif\n")
    f.close()

    # Open a file to define the problem data
    f = open(problem_name + "/" + problem_name + "_" + "data.cpp", "w")

    # Write include headers
    f.write('#include \"%s_data.h\"\n' % problem_name)
    f.write("\n\n")

    #
    # Generate QP problem data
    #
    f.write("/* Function to generate QP problem data */\n")
    f.write("%s_prob_data::%s_prob_data() : OSQPTestData() {\n" % (problem_name, problem_name))

    # Write problem dimensions
    f.write("// Problem dimensions\n")
    write_int(f, n, "n", "this")
    write_int(f, m, "m", "this")
    f.write("\n")

    # Write problem vectors
    f.write("// Problem vectors\n")
    write_vec_float(f, l, "l", "this")
    write_vec_float(f, u, "u", "this")
    write_vec_float(f, q, "q", "this")
    f.write("\n")

    # Write matrix A
    write_mat_sparse(f, A, "A", "this")
    write_mat_sparse(f, P, "P", "this")

    f.write("}\n\n")


    #
    # Generate additional problem data for solutions
    #
    f.write("/* Function to define solutions and additional data struct */\n")
    f.write("%s_sols_data::%s_sols_data() {\n" % (problem_name, problem_name))

    # Generate further data and solutions
    for key, value in sols_data.items():
        if isinstance(value, str):
            # Status test get from C code
            if value == 'optimal':
                f.write("%s = %s;\n" % (key, 'OSQP_SOLVED'))
            elif value == 'optimal_inaccurate':
                f.write("%s = %s;\n" % (key, 'OSQP_SOLVED_INACCURATE'))
            elif value == 'primal_infeasible':
                f.write("%s = %s;\n" % (key, 'OSQP_PRIMAL_INFEASIBLE'))
            elif value == 'primal_infeasible_inaccurate':
                f.write("%s = %s;\n" %
                        (key, 'OSQP_PRIMAL_INFEASIBLE_INACCURATE'))
            elif value == 'dual_infeasible':
                f.write("%s = %s;\n" % (key, 'OSQP_DUAL_INFEASIBLE'))
            elif value == 'dual_infeasible_inaccurate':
                f.write("%s = %s;\n" % (key, 'OSQP_DUAL_INFEASIBLE_INACCURATE'))

        # Check if it is an array or a scalar
        if type(value) is np.ndarray:
            if isinstance(value.flatten(order='F')[0], int):
                write_vec_int(f, value.flatten(order='F'), key, "this")
            elif isinstance(value.flatten(order='F')[0], float):
                write_vec_float(f, value.flatten(order='F'), key, "this")
        else:
            if isinstance(value, int):
                write_int(f, value, key, "this")
            elif isinstance(value, float):
                write_float(f, value, key, "this")

    # End function
    f.write("}\n\n")



    #
    # Clean additional problem data for solutions
    #
    f.write("/* Function to clean solutions and additional data struct */\n")
    f.write("%s_sols_data::~%s_sols_data() {\n" % (problem_name, problem_name))
    # Generate further data and solutions
    for key, value in sols_data.items():
        # Check if it is an array or a scalar
        if type(value) is np.ndarray:
            clean_vec(f, key)

    f.write("}\n\n")

    f.close()


def generate_data(problem_name, sols_data):
    """
    Generate test data vectors.

    The additional structure sols_data defines the additional vectors/scalars
    we need to perform the tests
    """

    #
    # GENERATE HEADER FILE
    #
    f = open(problem_name + "/" + problem_name + "_" + "data.h", "w")

    # Add definition check
    f.write("#ifndef " + problem_name.upper() + "_DATA_H\n")
    f.write("#define " + problem_name.upper() + "_DATA_H\n")

    # Add Includes
    f.write("#include \"osqp_api.h\"\n")
    f.write("#include \"osqp_tester.h\"\n")
    f.write("\n\n")

    #
    # Create additional data structure
    #
    f.write("/* create data and solutions structure */\n")
    f.write("typedef struct {\n")
    # Generate further data and solutions
    for key, value in sols_data.items():
        if isinstance(value, str):
            # Status test get from C code
            f.write("OSQPInt %s;\n" % key)
        # Check if it is an array or a scalar
        elif sparse.issparse(value):  # Sparse matrix
            f.write("OSQPCscMatrix* %s;\n" % key)
        elif isinstance(value, np.ndarray):
            if value.flatten(order='F').size == 0:
                f.write("OSQPFloat* %s;\n" % key)
            elif isinstance(value.flatten(order='F')[0], np.integer):
                f.write("OSQPInt* %s;\n" % key)
            elif isinstance(value.flatten(order='F')[0], np.single):
                f.write("OSQPFloat* %s;\n" % key)
            elif isinstance(value.flatten(order='F')[0], np.double):
                f.write("OSQPFloat* %s;\n" % key)
        else:
            if isinstance(value, int):
                f.write("OSQPInt %s;\n" % key)
            elif isinstance(value, float):
                f.write("OSQPFloat %s;\n" % key)
    f.write("} %s_sols_data;\n\n" % problem_name)

    # prototypes
    f.write("/* function prototypes */\n")
    f.write("%s_sols_data *  generate_problem_%s_sols_data();\n" % (problem_name, problem_name))
    f.write("void clean_problem_%s_sols_data(%s_sols_data * data);\n" % (problem_name, problem_name))
    f.write("\n\n")

    # Generate helpers for C++ memory management
    f.write("/* C++ memory management helpers */\n")
    f.write("#ifdef __cplusplus\n")
    f.write("#include <memory>\n\n")
    f.write("struct %s_sols_deleter {\n" % problem_name)
    f.write("    void operator()(%s_sols_data* sols_data) {\n" % problem_name)
    f.write("        clean_problem_%s_sols_data(sols_data);\n" % problem_name)
    f.write("    }\n")
    f.write("};\n\n")
    f.write("using %s_sols_data_ptr = std::unique_ptr<%s_sols_data, %s_sols_deleter>;\n" % (problem_name, problem_name, problem_name))
    f.write("#endif /* __cplusplus */\n\n")

    # Close header file
    f.write("#endif\n")
    f.close()

    # Open a file to define the problem data
    f = open(problem_name + "/" + problem_name + "_" + "data.cpp", "w")

    # Write include headers
    f.write('#include \"%s_data.h\"\n' % problem_name)
    f.write("\n\n")

    #
    # Generate additional problem data for solutions
    #
    f.write("/* function to define problem data */\n")
    f.write("%s_sols_data *  generate_problem_%s_sols_data(){\n\n" % (problem_name, problem_name))

    # Initialize structure data
    f.write("%s_sols_data * data = (%s_sols_data *)c_malloc(sizeof(%s_sols_data));\n\n" % (problem_name, problem_name, problem_name))

    # Generate further data and solutions
    for key, value in sols_data.items():
        if isinstance(value, str):
            # Status test get from C code
            if value == 'optimal':
                f.write("data->%s = %s;\n" % (key, 'OSQP_SOLVED'))
            elif value == 'primal_infeasible':
                f.write("data->%s = %s;\n" % (key, 'OSQP_PRIMAL_INFEASIBLE'))
            elif value == 'dual_infeasible':
                f.write("data->%s = %s;\n" % (key, 'OSQP_DUAL_INFEASIBLE'))
        # Check if it is an array or a scalar
        elif sparse.issparse(value):  # Sparse matrix
            write_mat_sparse(f, value, key, "data")
        elif type(value) is np.ndarray:
            # If the vector is empty, default to a float vector
            if value.flatten(order='F').size == 0:
                write_vec_float(f, value.flatten(order='F'), key, "data")
            elif isinstance(value.flatten(order='F')[0], np.integer):
                write_vec_int(f, value.flatten(order='F'), key, "data")
            elif isinstance(value.flatten(order='F')[0], np.single):
                write_vec_float(f, value.flatten(order='F'), key, "data")
            elif isinstance(value.flatten(order='F')[0], np.double):
                write_vec_float(f, value.flatten(order='F'), key, "data")
        else:
            if isinstance(value, int):
                write_int(f, value, key, "data")
            elif isinstance(value, float):
                write_float(f, value, key, "data")

    # Return data and end function
    f.write("\nreturn data;\n\n")

    f.write("}\n\n")


    #
    # Clean  data
    #
    f.write("/* function to clean data struct */\n")
    f.write("void clean_problem_%s_sols_data(%s_sols_data * data){\n\n" % (problem_name, problem_name))
    # Generate further data and solutions
    for key, value in sols_data.items():
        # Check if it is an array or a scalar
        if sparse.issparse(value):  # Sparse matrix
            clean_mat(f, key, "data")
        elif type(value) is np.ndarray:
            clean_vec(f, key, "data")

    f.write("\nc_free(data);\n\n")

    f.write("}\n\n")

    f.close()
