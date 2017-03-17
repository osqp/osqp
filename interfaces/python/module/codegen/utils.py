"""
Utilities to generate embedded C code from OSQP sources
"""
# Compatibility with Python 2
from __future__ import print_function
from builtins import range

# Import numpy
import numpy as np

# Path of osqp module
import os.path
import osqp
files_to_generate_path = os.path.join(osqp.__path__[0],
                                      'codegen', 'files_to_generate')

# def fill_scalar(f, scalar, name, scal_type):
#     """
#     Fill scalar value
#     """
#     f.write("%s = " % (name))
#     if scal_type == 'c_float':
#         f.write("(c_float)%.20f" % scalar)
#     else:
#         f.write("%i" % scalar)
#     f.write(";\n")
#
# def fill_vec(f, vec, name, vec_type):
#     """
#     Fill numpy array
#     """
#     for i in range(vec.size):
#         fill_scalar(f, vec[i], "%s[%i]" % (name, i), vec_type)
#
#
# def fill_mat(f, mat, name):
#     """
#     Scipy sparse csc matrix
#     """
#     fill_vec(f, mat['i'], name+'->i', 'c_int')
#     fill_vec(f, mat['p'], name+'->p', 'c_int')
#     fill_vec(f, mat['x'], name+'->x', 'c_float')
#     fill_scalar(f, mat['m'], "%s->m" % (name), 'c_int')
#     fill_scalar(f, mat['n'], "%s->n" % (name), 'c_int')
#     fill_scalar(f, mat['nz'], "%s->nz" % (name), 'c_int')
#     fill_scalar(f, mat['nzmax'], "%s->nzmax" % (name), 'c_int')



def fill_settings(f, settings, name):
    """
    Fill settings
    """
    for key, value in settings.items():
        if key != 'scaling_norm' and key != 'scaling_iter':
            if type(value) == int:
                is_int = True
            elif value.is_integer():
                is_int = True
            else:
                is_int = False
            f.write('%s->%s = ' % (name, key))
            if is_int:
                f.write(str(value))
            else:
                f.write("(c_float)")
                f.write(str(value))
            f.write(";\n")


def write_vec(f, vec, name, vec_type):
    """
    Write vector to file
    """
    f.write('%s %s[%d] = {\n' % (vec_type, name, len(vec)))

    # Write vector elements
    for i in range(len(vec)):
        if vec_type == 'c_float':
            f.write('%.20f,\n' % vec[i])
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
    f.write(".nzmax = %d, " % mat['nzmax'])
    f.write(".m = %d, " % mat['m'])
    f.write(".n = %d, " % mat['n'])
    f.write(".p = %s_p, " % name)
    f.write(".i = %s_i, " % name)
    f.write(".x = %s_x, " % name)
    f.write(".nz = %d};\n" % mat['nz'])


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
    f.write(".n = %d, " % data['n'])
    f.write(".m = %d, " % data['m'])
    f.write(".P = &Pdata, .A = &Adata, .q = qdata, .l = ldata, .u = udata")
    f.write("};\n\n")



def write_settings(f, settings, name):
    """
    Fille settings during code generation
    """
    f.write("// Define settings structure\n")
    f.write("OSQPSettings %s = {" % name)
    f.write(".rho = %.20f, " % settings['rho'])
    f.write(".sigma = %.20f, " % settings['sigma'])
    f.write(".scaling = %d, " % settings['scaling'])

    # TODO: Add scaling_norm and scaling_iter for EMBEDDED = 2

    f.write(".max_iter = %d, " % settings['max_iter'])
    f.write(".eps_abs = %.20f, " % settings['eps_abs'])
    f.write(".eps_rel = %.20f, " % settings['eps_rel'])
    f.write(".eps_inf = %.20f, " % settings['eps_inf'])
    f.write(".eps_unb = %.20f, " % settings['eps_unb'])
    f.write(".alpha = %.20f, " % settings['alpha'])

    f.write(".early_terminate = %d, " % settings['early_terminate'])
    f.write(".early_terminate_interval = %d, " %
            settings['early_terminate_interval'])
    f.write(".warm_start = %d" % settings['warm_start'])

    f.write("};\n\n")


def write_scaling(f, scaling, name):
    """
    Write scaling structure during code generation
    """
    f.write("// Define scaling structure\n")
    write_vec(f, scaling['D'], 'Dscaling', 'c_float')
    write_vec(f, scaling['Dinv'], 'Dinvscaling', 'c_float')
    write_vec(f, scaling['E'], 'Escaling', 'c_float')
    write_vec(f, scaling['Einv'], 'Einvscaling', 'c_float')
    f.write("OSQPScaling %s = " % name)
    f.write("{.D = Dscaling, .E = Escaling, .Dinv = Dinvscaling," +
            " .Einv = Einvscaling};\n\n")


def write_private(f, priv, name):
    """
    Write private structure during code generation
    """

    f.write("// Define private structure\n")
    write_mat(f, priv['L'], 'priv_L')
    write_vec(f, priv['Dinv'], 'priv_Dinv', 'c_float')
    write_vec(f, priv['P'], 'priv_P', 'c_int')
    f.write("c_float priv_bp[%d];\n" % (len(priv['Dinv'])))  # Empty rhs

    f.write("Priv %s = " % name)
    f.write("{.L = &priv_L, .Dinv = priv_Dinv," +
            " .P = priv_P, .bp = priv_bp};\n\n")


def write_solution(f, data, name):
    """
    Preallocate solution vectors
    """
    f.write("// Define solution\n")
    f.write("c_float xsolution[%d];\n" % data['n'])
    f.write("c_float ysolution[%d];\n\n" % data['m'])
    f.write("OSQPSolution %s = {.x = xsolution, .y = ysolution};\n\n" % name)


def write_info(f, name):
    """
    Preallocate info strcture
    """
    f.write("// Define info\n")
    f.write("OSQPInfo %s = {.status_val = OSQP_UNSOLVED};\n\n" % name)


def write_workspace(f, data, name):
    """
    Write workspace structure
    """

    f.write("// Define workspace\n")
    f.write("c_float work_x[%d];\n" % data['n'])
    f.write("c_float work_y[%d];\n" % data['m'])
    f.write("c_float work_z[%d];\n" % data['m'])
    f.write("c_float work_xz_tilde[%d];\n" % (data['m'] + data['n']))
    f.write("c_float work_x_prev[%d];\n" % data['n'])
    f.write("c_float work_z_prev[%d];\n" % data['m'])
    f.write("c_float work_delta_y[%d];\n" % data['m'])
    f.write("c_float work_Atdelta_y[%d];\n" % data['n'])
    f.write("c_float work_delta_x[%d];\n" % data['n'])
    f.write("c_float work_Pdelta_x[%d];\n" % data['n'])
    f.write("c_float work_Adelta_x[%d];\n" % data['m'])
    f.write("c_float work_P_x[%d];\n" % data['n'])
    f.write("c_float work_A_x[%d];\n" % data['m'])
    f.write("c_float work_D_temp[%d];\n" % data['n'])
    f.write("c_float work_E_temp[%d];\n\n" % data['m'])

    f.write("OSQPWorkspace %s = {\n" % name)
    f.write(".data = &data, .priv = &priv,\n")
    f.write(".x = work_x, .y = work_y, .z = work_z," +
            " .xz_tilde = work_xz_tilde,\n")
    f.write(".x_prev = work_x_prev, .z_prev = work_z_prev,\n")
    f.write(".delta_y = work_delta_y, .Atdelta_y = work_Atdelta_y,\n")
    f.write(".delta_x = work_delta_x, .Pdelta_x = work_Pdelta_x, " +
            ".Adelta_x = work_Adelta_x,\n")
    f.write(".P_x = work_P_x, .A_x = work_A_x,\n")
    f.write(".D_temp = work_D_temp, .E_temp = work_E_temp,\n")
    f.write(".settings = &settings, .scaling = &scaling, " +
            ".solution = &solution, .info = &info};\n\n")


def render_workspace(variables, output):
    """
    Print workspace dimensions
    """

    data = variables['data']
    priv = variables['priv']
    scaling = variables['scaling']
    settings = variables['settings']

    # Open output file
    f = open(output, 'w')

    # Include types, constants and private header
    f.write("#include \"types.h\"\n")
    f.write("#include \"constants.h\"\n")
    f.write("#include \"private.h\"\n\n")

    # Redefine type of structure in private
    f.write("// Redefine type of the structure in private\n")
    f.write("// N.B. Making sure the right amount of memory is allocated\n")
    f.write("typedef struct c_priv Priv;\n\n")

    '''
    Write data structure
    '''
    write_data(f, data, 'data')

    '''
    Write settings structure
    '''
    write_settings(f, settings, 'settings')

    '''
    Write scaling structure
    '''
    write_scaling(f, scaling, 'scaling')

    '''
    Write private structure
    '''
    write_private(f, priv, 'priv')

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
    write_workspace(f, data, 'workspace')

    f.close()

    # import ipdb; ipdb.set_trace()
    # Write load workspace function
    #
    # f.write("// Populate workspace\n")
    # f.write("void load_workspace(OSQPWorkspace * work){\n\n")
    #
    # f.write("// Link structures\n")
    # f.write("link_structures();\n\n")

    # f.write("/* Fill data */\n\n")

    # f.write("// Data dimensions\n\n")


    # fill_scalar(f, data['n'], "work->data->n", 'c_int')
    # fill_scalar(f, data['m'], "work->data->m", 'c_int')
    # f.write("\n")


    # f.write("// Fill P\n")
    # fill_mat(f, data['P'], 'work->data->P')
    # f.write("\n")

    # f.write("// Fill q\n")
    # fill_vec(f, data['q'], 'work->data->q', 'c_float')
    # f.write("\n")
    #
    # f.write("// Fill A\n")
    # fill_mat(f, data['A'], 'work->data->A')
    # f.write("\n")
    #
    # f.write("// Fill l\n")
    # fill_vec(f, data['l'], 'work->data->l', 'c_float')
    # f.write("\n")
    #
    # f.write("// Fill u\n")
    # fill_vec(f, data['u'], 'work->data->u', 'c_float')
    # f.write('\n')
    #
    #
    # f.write("/* Fill settings */\n")
    # fill_settings(f, settings, 'work->settings')
    # f.write("\n")

    # f.write("/* Fill scaling */\n")
    # fill_vec(f, scaling['D'], 'work->scaling->D', 'c_float')
    # fill_vec(f, scaling['Dinv'], 'work->scaling->Dinv', 'c_float')
    # fill_vec(f, scaling['E'], 'work->scaling->E', 'c_float')
    # fill_vec(f, scaling['Einv'], 'work->scaling->Einv', 'c_float')
    # f.write("\n")

    # f.write("/* Fill private structure */\n")
    # fill_vec(f, priv['P'], 'work->priv->P', 'c_int')
    # fill_vec(f, priv['Dinv'], 'work->priv->Dinv', 'c_float')
    # fill_mat(f, priv['L'], 'work->priv->L')
    # f.write("\n")

    # f.write("/* Set info */\n")
    # f.write("work->info->status_val = OSQP_UNSOLVED;\n\n")


    # f.write("}\n")

    # Close file
    # f.close()

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
