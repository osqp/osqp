# from osqp import __path__
import osqp
from jinja2 import Environment, PackageLoader, contextfilter
import os.path
import shutil as sh
from subprocess import call


# PKG_DIR = os.path.dirname(os.path.abspath(__path__[0]))

# OSQP_DIR = os.path.join(PKG_DIR, 'osqp', 'codegen', 'osqp')
# OSQP_SRC_DIR = os.path.join(OSQP_DIR, 'src')
# OSQP_INCLUDE_DIR = os.path.join(OSQP_DIR, 'include')
# OSQP_LINSYS_DIR = os.path.join(OSQP_DIR, 'lin_sys')
# OSQP_MAKEFILE = os.path.join(OSQP_DIR, 'Makefile')


def render(target_dir, template_vars, template_name, target_name):
    # total_template_vars = dict()
    # total_template_vars.update(template_vars)
    #total_template_vars.update(DEFAULT_TEMPLATE_VARS) # TODO

    env = Environment(loader=PackageLoader('osqp.codegen', 'jinja'),
                      lstrip_blocks=True,
                      trim_blocks=True)

    template = env.get_template(template_name)
    f = open(os.path.join(target_dir, target_name), 'w')
    f.write(template.render(template_vars))
    f.close()



def codegen(work, target_dir, project_type, embedded_flag):
    """
    Generate code
    """

    # Import OSQP path
    osqp_path = osqp.__path__[0]


    # Make target directory
    target_dir = os.path.abspath(target_dir)
    target_include_dir = os.path.join(target_dir, 'include')
    target_src_dir = os.path.join(target_dir, 'src')

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(target_include_dir):
        os.mkdir(target_include_dir)
    if not os.path.exists(target_src_dir):
        os.mkdir(target_src_dir)



    # Define target include directory (where to put headers)




    # Copy source files to target directory
    #
    #






    # if os.path.exists(target_dir):
    #     sh.rmtree(target_dir)  # TODO a bit heavy handed..
    # sh.copytree(OSQP_DIR, target_dir)



    # Make subdirectories
    # target_src_dir = os.path.join(target_dir, 'src')

    #target_linsys_dir = os.path.join(target_dir, 'lin_sys')
    #target_cmake_dir = os.path.join(target_dir, 'CMakeFiles')

    #if os.path.exists(target_src_dir):
    #    sh.rmtree(target_src_dir)  # TODO a bit heavy handed..
    #if os.path.exists(target_include_dir):
    #    sh.rmtree(target_include_dir)
    #if os.path.exists(target_linsys_dir):
    #    sh.rmtree(target_linsys_dir)
    #if os.path.exists(target_cmake_dir):
    #    sh.rmtree(target_cmake_dir)


    # Copy OSQP source files
    #sh.copytree(OSQP_SRC_DIR, target_src_dir)
    #sh.copytree(OSQP_SRC_DIR, target_src_dir)
    #sh.copytree(OSQP_INCLUDE_DIR, target_include_dir)
    #sh.copytree(OSQP_LINSYS_DIR, target_linsys_dir)
    #sh.copytree(os.path.join(OSQP_DIR, 'CMakeFiles'), target_cmake_dir)
    #sh.copy(OSQP_MAKEFILE, target_dir)
    #sh.copy(OSQP_OSQPMK, target_dir)

    template_vars = {'data'     : work['data'],
                     'settings' : work['settings'],
                     'priv'     : work['priv'],
                     'scaling'  : work['scaling'],
                     'embedded_flag': embedded_flag}

    import ipdb; ipdb.set_trace()
    render(target_include_dir, template_vars, 'workspace.h.jinja', 'workspace.h')



    # Generate project
    # cwd = os.getcwd()
    # os.chdir(target_dir)
    # call(["cmake", "-DEMBEDDED=%i" % embedded_flag, ".."])
    # os.chdir(cwd)

    #render(target_src_dir, template_vars, 'osqp_cg_data.c.jinja',
    #       'osqp_cg_data.c')
    #render(target_dir, template_vars, 'example_problem.c.jinja',
    #       'example_problem.c')
