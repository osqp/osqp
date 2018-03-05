"""
Utilities to generate embedded C code from OSQP sources
"""
# Compatibility with Python 2
from __future__ import print_function
from builtins import range

# Path of osqp module
import os.path
import osqp
files_to_generate_path = os.path.join(osqp.__path__[0],
                                      'codegen', 'files_to_generate')


def write_vec(f, vec, name, vec_type):
    """
    Write vector to file
    """
    f.write('%s %s[%d] = {\n' % (vec_type, name, len(vec)))

    # Write vector elements
    for i in range(len(vec)):
        if vec_type == 'c_float':
            f.write('(c_float)%.20f,\n' % vec[i])
        else:
            f.write('%i,\n' % vec[i])

    f.write('};\n')


def write_mat(f, mat, name):
    """
    Write scipy sparse matrix in CSC form to file
    """
    write_vec(f, mat['i'], name + '_i', 'c_int')
    write_vec(f, mat['p'], name + '_p', 'c_int')
    write_vec(f, mat['x'], name + '_x', 'c_float')

    # import ipdb; ipdb.set_trace()

    f.write("csc %s = {" % name)
    f.write("%d, " % mat['nzmax'])
    f.write("%d, " % mat['m'])
    f.write("%d, " % mat['n'])
    f.write("%s_p, " % name)
    f.write("%s_i, " % name)
    f.write("%s_x, " % name)
    f.write("%d};\n" % mat['nz'])


def write_data(f, data, name):
    """
    Write data structure during code generation
    """

    f.write("// Define data structure\n")

    # Define matrix P
    write_mat(f, data['P'], 'Pdata')

    # Define matrix A
    write_mat(f, data['A'], 'Adata')

    # Define other data vectors
    write_vec(f, data['q'], 'qdata', 'c_float')
    write_vec(f, data['l'], 'ldata', 'c_float')
    write_vec(f, data['u'], 'udata', 'c_float')

    # Define data structure
    f.write("OSQPData data = {")
    f.write("%d, " % data['n'])
    f.write("%d, " % data['m'])
    f.write("&Pdata, &Adata, qdata, ldata, udata")
    f.write("};\n\n")


def write_settings(f, settings, name, embedded_flag):
    """
    Fille settings during code generation
    """
    f.write("// Define settings structure\n")
    f.write("OSQPSettings %s = {" % name)
    f.write("(c_float)%.20f, " % settings['rho'])
    f.write("(c_float)%.20f, " % settings['sigma'])
    f.write("%d, " % settings['scaling'])

    if embedded_flag != 1:
        f.write("%d, " % settings['adaptive_rho'])
        f.write("%d, " % settings['adaptive_rho_interval'])
        f.write("(c_float)%.20f, " % settings['adaptive_rho_tolerance'])
        f.write("\n#ifdef PROFILING\n")
        f.write("(c_float)%.20f, " % settings['adaptive_rho_fraction'])
        f.write("\n#endif  // PROFILING\n")

    f.write("%d, " % settings['max_iter'])
    f.write("(c_float)%.20f, " % settings['eps_abs'])
    f.write("(c_float)%.20f, " % settings['eps_rel'])
    f.write("(c_float)%.20f, " % settings['eps_prim_inf'])
    f.write("(c_float)%.20f, " % settings['eps_dual_inf'])
    f.write("(c_float)%.20f, " % settings['alpha'])
    f.write("%d, " % settings['linsys_solver'])

    f.write("%d, " % settings['scaled_termination'])
    f.write("%d, " % settings['check_termination'])
    f.write("%d," % settings['warm_start'])


    f.write("\n#ifdef PROFILING\n")
    f.write("(c_float)%.20f " % settings['time_limit'])
    f.write("\n#endif  // PROFILING\n")

    f.write("};\n\n")


def write_scaling(f, scaling, name):
    """
    Write scaling structure during code generation
    """
    f.write("// Define scaling structure\n")
    if scaling is not None:
        write_vec(f, scaling['D'], 'Dscaling', 'c_float')
        write_vec(f, scaling['Dinv'], 'Dinvscaling', 'c_float')
        write_vec(f, scaling['E'], 'Escaling', 'c_float')
        write_vec(f, scaling['Einv'], 'Einvscaling', 'c_float')
        f.write("OSQPScaling %s = {" % name)
        f.write("(c_float)%.20f, " % scaling['c'])
        f.write("Dscaling, Escaling, ")
        f.write("(c_float)%.20f, " % scaling['cinv'])
        f.write("Dinvscaling, Einvscaling};\n\n")
    else:
        f.write("OSQPScaling %s;\n\n" % name)


def write_linsys_solver(f, linsys_solver, name, embedded_flag):
    """
    Write linsys_solver structure during code generation
    """

    f.write("// Define linsys_solver structure\n")
    write_mat(f, linsys_solver['L'], 'linsys_solver_L')
    write_vec(f, linsys_solver['Dinv'], 'linsys_solver_Dinv', 'c_float')
    write_vec(f, linsys_solver['P'], 'linsys_solver_P', 'c_int')
    # Empty rhs
    f.write("c_float linsys_solver_bp[%d];\n" % (len(linsys_solver['Dinv'])))

    if embedded_flag != 1:
        write_vec(f, linsys_solver['Pdiag_idx'],
                  'linsys_solver_Pdiag_idx', 'c_int')
        write_mat(f, linsys_solver['KKT'], 'linsys_solver_KKT')
        write_vec(f, linsys_solver['PtoKKT'], 'linsys_solver_PtoKKT', 'c_int')
        write_vec(f, linsys_solver['AtoKKT'], 'linsys_solver_AtoKKT', 'c_int')
        write_vec(f, linsys_solver['rhotoKKT'],
                  'linsys_solver_rhotoKKT', 'c_int')
        write_vec(f, linsys_solver['Lnz'], 'linsys_solver_Lnz', 'c_int')
        write_vec(f, linsys_solver['Y'], 'linsys_solver_Y', 'c_float')
        write_vec(f, linsys_solver['Pattern'],
                  'linsys_solver_Pattern', 'c_int')
        write_vec(f, linsys_solver['Flag'], 'linsys_solver_Flag', 'c_int')
        write_vec(f, linsys_solver['Parent'], 'linsys_solver_Parent', 'c_int')

    f.write("suitesparse_ldl_solver %s = " % name)
    f.write("{SUITESPARSE_LDL_SOLVER, &solve_linsys_suitesparse_ldl, ")
    if embedded_flag != 1:
        f.write("&update_linsys_solver_matrices_suitesparse_ldl, &update_linsys_solver_rho_vec_suitesparse_ldl, " +
                "&linsys_solver_L, linsys_solver_Dinv, linsys_solver_P, linsys_solver_bp, linsys_solver_Pdiag_idx, " +
                "%d, " % linsys_solver['Pdiag_n'] +
                "&linsys_solver_KKT, linsys_solver_PtoKKT, linsys_solver_AtoKKT, linsys_solver_rhotoKKT, " +
                "linsys_solver_Lnz, linsys_solver_Y, linsys_solver_Pattern, linsys_solver_Flag, linsys_solver_Parent};\n\n")
    else:
        f.write("&linsys_solver_L, linsys_solver_Dinv, linsys_solver_P, linsys_solver_bp};\n\n")


def write_solution(f, data, name):
    """
    Preallocate solution vectors
    """
    f.write("// Define solution\n")
    f.write("c_float xsolution[%d];\n" % data['n'])
    f.write("c_float ysolution[%d];\n\n" % data['m'])
    f.write("OSQPSolution %s = {xsolution, ysolution};\n\n" % name)


def write_info(f, name):
    """
    Preallocate info strcture
    """
    f.write("// Define info\n")
    f.write('OSQPInfo %s = {0, "Unsolved", OSQP_UNSOLVED, 0.0, 0.0, 0.0};\n\n'
            % name)


def write_workspace(f, n, m, rho_vectors, embedded_flag, name):
    """
    Write workspace structure
    """

    f.write("// Define workspace\n")

    write_vec(f, rho_vectors['rho_vec'], 'work_rho_vec', 'c_float')
    write_vec(f, rho_vectors['rho_inv_vec'], 'work_rho_inv_vec', 'c_float')
    if embedded_flag != 1:
        write_vec(f, rho_vectors['constr_type'], 'work_constr_type', 'c_int')

    f.write("c_float work_x[%d];\n" % n)
    f.write("c_float work_y[%d];\n" % m)
    f.write("c_float work_z[%d];\n" % m)
    f.write("c_float work_xz_tilde[%d];\n" % (m + n))
    f.write("c_float work_x_prev[%d];\n" % n)
    f.write("c_float work_z_prev[%d];\n" % m)
    f.write("c_float work_Ax[%d];\n" % m)
    f.write("c_float work_Px[%d];\n" % n)
    f.write("c_float work_Aty[%d];\n" % n)
    f.write("c_float work_delta_y[%d];\n" % m)
    f.write("c_float work_Atdelta_y[%d];\n" % n)
    f.write("c_float work_delta_x[%d];\n" % n)
    f.write("c_float work_Pdelta_x[%d];\n" % n)
    f.write("c_float work_Adelta_x[%d];\n" % m)
    f.write("c_float work_D_temp[%d];\n" % n)
    f.write("c_float work_D_temp_A[%d];\n" % n)
    f.write("c_float work_E_temp[%d];\n\n" % m)

    f.write("OSQPWorkspace %s = {\n" % name)
    f.write("&data, (LinSysSolver *)&linsys_solver,\n")
    f.write("work_rho_vec, work_rho_inv_vec,\n")
    if embedded_flag != 1:
        f.write("work_constr_type,\n")

    f.write("work_x, work_y, work_z, work_xz_tilde,\n")
    f.write("work_x_prev, work_z_prev,\n")
    f.write("work_Ax, work_Px, work_Aty,\n")
    f.write("work_delta_y, work_Atdelta_y,\n")
    f.write("work_delta_x, work_Pdelta_x, work_Adelta_x,\n")
    f.write("work_D_temp, work_D_temp_A, work_E_temp,\n")
    f.write("&settings, &scaling, &solution, &info};\n\n")


def render_workspace(variables, output):
    """
    Print workspace dimensions
    """

    rho_vectors = variables['rho_vectors']
    data = variables['data']
    linsys_solver = variables['linsys_solver']
    scaling = variables['scaling']
    settings = variables['settings']
    embedded_flag = variables['embedded_flag']
    n = data['n']
    m = data['m']

    # Open output file
    f = open(output, 'w')

    # Include types, constants and linsys_solver header
    f.write("#include \"types.h\"\n")
    f.write("#include \"constants.h\"\n")
    f.write("#include \"suitesparse_ldl.h\"\n\n")

    '''
    Write data structure
    '''
    write_data(f, data, 'data')

    '''
    Write settings structure
    '''
    write_settings(f, settings, 'settings', embedded_flag)

    '''
    Write scaling structure
    '''
    write_scaling(f, scaling, 'scaling')

    '''
    Write linsys_solver structure
    '''
    write_linsys_solver(f, linsys_solver, 'linsys_solver', embedded_flag)

    '''
    Define empty solution structure
    '''
    write_solution(f, data, 'solution')

    '''
    Define info structure
    '''
    write_info(f, 'info')

    '''
    Define workspace structure
    '''
    write_workspace(f, n, m, rho_vectors, embedded_flag, 'workspace')

    f.close()


def write_ldl_lsolve(f, variables):
    """
    Write LDL_lsolve to file
    """

    data = variables['data']
    linsys_solver = variables['linsys_solver']
    Lp = linsys_solver['L']['p']

    f.write("void LDL_lsolve(LDL_int n, c_float X [ ], LDL_int Lp [ ]")
    f.write(", LDL_int Li [ ], c_float Lx [ ]){\n")
    f.write("LDL_int p;\n")

    # Unroll for loop
    for j in range(data['m'] + data['n']):
        if Lp[j+1] > Lp[j]:  # Write loop ONLY if necessary
            f.write("for (p = %i ; p < %i ; p++){\n" % (Lp[j], Lp[j+1]))
            f.write("X [Li [p]] -= Lx [p] * X [%i];\n" % (j))
            f.write("}\n")

    # Close function
    f.write("}\n\n")


def write_ldl_ltsolve(f, variables):
    """
    Write LDL_ltsolve to file
    """
    data = variables['data']
    linsys_solver = variables['linsys_solver']
    Lp = linsys_solver['L']['p']

    f.write("void LDL_ltsolve(LDL_int n, c_float X [ ], LDL_int Lp [ ]")
    f.write(", LDL_int Li [ ], c_float Lx [ ]){\n")
    f.write("LDL_int p;\n")

    # Unroll the loop
    for j in range(data['m'] + data['n'] - 1, -1, -1):
        if Lp[j+1] > Lp[j]:  # Write loop ONLY if necessary
            f.write("for (p = %i ; p < %i ; p++){\n" % (Lp[j], Lp[j+1]))
            f.write("X [%i] -= Lx [p] * X [Li [p]] ;\n" % (j))
            f.write("}\n")

    # Close function
    f.write("}\n\n")


def write_ldl_dinvsolve(f, variables):
    """
    Write LDL_dinvsolve
    """
    data = variables['data']

    f.write("void LDL_dinvsolve(LDL_int n, c_float X [ ], ")
    f.write("c_float Dinv [ ]) {\n")
    f.write("LDL_int i;\n")
    f.write("for (i = 0 ; i < %i ; i++){\n" % (data['m'] + data['n']))
    f.write("X[i] *= Dinv[i];\n")
    f.write("}\n")

    # Close function
    f.write("}\n\n")


def write_ldl_perm(f, variables):
    """
    Write LDL_perm
    """
    data = variables['data']

    f.write("void LDL_perm(LDL_int n, c_float X [ ], c_float B [ ],")
    f.write(" LDL_int P [ ]){\n")
    f.write("LDL_int j;\n")

    f.write("for (j = 0 ; j < %i ; j++){\n" % (data['m'] + data['n']))
    f.write("X [j] = B [P [j]];\n")
    f.write("}\n")

    # Close function
    f.write("}\n\n")


def write_ldl_permt(f, variables):
    """
    Write LDL_permt
    """
    data = variables['data']

    f.write("void LDL_permt(LDL_int n, c_float X [ ], c_float B [ ],")
    f.write(" LDL_int P [ ]){\n")
    f.write("LDL_int j;\n")

    f.write("for (j = 0 ; j < %i ; j++){\n" % (data['m'] + data['n']))
    f.write("X [P [j]] = B [j] ;\n")
    f.write("}\n")

    # Close function
    f.write("}\n\n")


def render_ldl(variables, output):
    """
    Render LDL file so that loops can be unrolled
    """

    f = open(output, 'w')

    # Include header
    f.write("#include \"ldl.h\"\n\n")

    # Write ldl_lsolve
    write_ldl_lsolve(f, variables)

    # Write ldl_ltsolve
    write_ldl_ltsolve(f, variables)

    # Write ldl_dinvsolve
    write_ldl_dinvsolve(f, variables)

    # Write ldl_perm
    write_ldl_perm(f, variables)

    # Write ldl_permt
    write_ldl_permt(f, variables)

    f.close()


def render_setuppy(variables, output):
    """
    Render setup.py file
    """

    embedded_flag = variables['embedded_flag']
    python_ext_name = variables['python_ext_name']

    f = open(os.path.join(files_to_generate_path, 'setup.py'))
    filedata = f.read()
    f.close()

    filedata = filedata.replace("EMBEDDED_FLAG", str(embedded_flag))
    filedata = filedata.replace("PYTHON_EXT_NAME", str(python_ext_name))

    f = open(output, 'w')
    f.write(filedata)
    f.close()


def render_cmakelists(variables, output):
    """
    Render CMakeLists file
    """

    embedded_flag = variables['embedded_flag']

    f = open(os.path.join(files_to_generate_path, 'CMakeLists.txt'))
    filedata = f.read()
    f.close()

    filedata = filedata.replace("EMBEDDED_FLAG", str(embedded_flag))

    f = open(output, 'w')
    f.write(filedata)
    f.close()


def render_emosqpmodule(variables, output):
    """
    Render emosqpmodule.c file
    """

    python_ext_name = variables['python_ext_name']

    f = open(os.path.join(files_to_generate_path, 'emosqpmodule.c'))
    filedata = f.read()
    f.close()

    filedata = filedata.replace("PYTHON_EXT_NAME", str(python_ext_name))

    f = open(output, 'w')
    f.write(filedata)
    f.close()
