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

def fill_scalar(f, scalar, name, scal_type):
    """
    Fill scalar value
    """
    f.write("%s = " % (name))
    if scal_type == 'c_float':
        f.write("(c_float)%.20f" % scalar)
    else:
        f.write("%i" % scalar)
    f.write(";\n")

def fill_vec(f, vec, name, vec_type):
    """
    Fill numpy array
    """
    for i in range(vec.size):
        fill_scalar(f, vec[i], "%s[%i]" % (name, i), vec_type)


def fill_mat(f, mat, name):
    """
    Scipy sparse csc matrix
    """
    fill_vec(f, mat['i'], name+'->i', 'c_int')
    fill_vec(f, mat['p'], name+'->p', 'c_int')
    fill_vec(f, mat['x'], name+'->x', 'c_float')
    fill_scalar(f, mat['m'], "%s->m" % (name), 'c_int')
    fill_scalar(f, mat['n'], "%s->n" % (name), 'c_int')
    fill_scalar(f, mat['nz'], "%s->nz" % (name), 'c_int')
    fill_scalar(f, mat['nzmax'], "%s->nzmax" % (name), 'c_int')


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



def render_setuppy(variables, output):
    """
    Render setup.py file
    """

    embedded_flag = variables['embedded_flag']

    f = open(os.path.join(files_to_generate_path, 'setup.py'))
    filedata = f.read()
    f.close()

    newdata = filedata.replace("EMBEDDED_FLAG", str(embedded_flag))

    f = open(output,'w')
    f.write(newdata)
    f.close()



def render_workspace(variables, output):
    """
    Print workspace dimensions
    """

    data = variables['data']
    priv = variables['priv']
    scaling = variables['scaling']
    settings = variables['settings']
    import ipdb; ipdb.set_trace()
    # embedded_flag = variables['embedded_flag']

    # Open output file
    f = open(output, 'w')

    f.write("#define OSQP_NDIM (%i)\n" % data['n'])
    f.write("#define OSQP_MDIM (%i)\n" % data['m'])
    f.write("#define OSQP_A_NNZ (%i)\n" % data['A']['nzmax'])
    f.write("#define OSQP_P_NNZ (%i)\n" % data['P']['nzmax'])
    f.write("#define OSQP_L_NNZ (%i)\n" % priv['L']['nzmax'])
    f.write("#define OSQP_KKT_NDIM (%i)\n" % (data['n'] + data['m']))
    f.write("\n\n")


    # Open partial workspace
    workspace_partial = open(os.path.join(files_to_generate_path, 'workspace_partial.h'))

    # Copy partial workspace into the output file
    f.write(workspace_partial.read())


    # Write load workspace function

    f.write("// Populate workspace\n")
    f.write("void load_workspace(OSQPWorkspace * work){\n\n")

    f.write("// Link structures\n")
    f.write("link_structures();\n\n")

    f.write("/* Fill data */\n\n")

    f.write("// Data dimensions\n\n")


    fill_scalar(f, data['n'], "work->data->n", 'c_int')
    fill_scalar(f, data['m'], "work->data->m", 'c_int')
    f.write("\n")


    f.write("// Fill P\n")
    fill_mat(f, data['P'], 'work->data->P')
    f.write("\n")

    f.write("// Fill q\n")
    fill_vec(f, data['q'], 'work->data->q', 'c_float')
    f.write("\n")

    f.write("// Fill A\n")
    fill_mat(f, data['A'], 'work->data->A')
    f.write("\n")

    f.write("// Fill l\n")
    fill_vec(f, data['l'], 'work->data->l', 'c_float')
    f.write("\n")

    f.write("// Fill u\n")
    fill_vec(f, data['u'], 'work->data->u', 'c_float')
    f.write('\n')


    f.write("/* Fill settings */\n")
    fill_settings(f, settings, 'work->settings')
    f.write("\n")

    f.write("/* Fill scaling */\n")
    fill_vec(f, scaling['D'], 'work->scaling->D', 'c_float')
    fill_vec(f, scaling['Dinv'], 'work->scaling->Dinv', 'c_float')
    fill_vec(f, scaling['E'], 'work->scaling->E', 'c_float')
    fill_vec(f, scaling['Einv'], 'work->scaling->Einv', 'c_float')
    f.write("\n")

    f.write("/* Fill private structure */\n")
    fill_vec(f, priv['P'], 'work->priv->P', 'c_int')
    fill_vec(f, priv['Dinv'], 'work->priv->Dinv', 'c_float')
    fill_mat(f, priv['L'], 'work->priv->L')
    f.write("\n")

    f.write("/* Set info */\n")
    f.write("work->info->status_val = OSQP_UNSOLVED;\n\n")


    f.write("}\n")

    # Close file
    f.close()
