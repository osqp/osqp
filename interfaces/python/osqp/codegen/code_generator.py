# from osqp import __path__
import osqp
from jinja2 import Environment, PackageLoader, contextfilter
import os.path
import shutil as sh
from subprocess import call
from glob import glob


def render(target_dir, template_vars, template_name, target_name):

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




    # Copy source files to target directory
    c_sources = glob(os.path.join(osqp_path, 'codegen','sources', 'src', '*.c'))
    for source in c_sources:
        sh.copy(source, target_src_dir)
    c_headers = glob(os.path.join(osqp_path, 'codegen','sources', 'include', '*.h'))
    for header in c_headers:
        sh.copy(header, target_include_dir)


    # Variables created from the workspace
    template_vars = {'data'     : work['data'],
                     'settings' : work['settings'],
                     'priv'     : work['priv'],
                     'scaling'  : work['scaling'],
                     'embedded_flag': embedded_flag}

    # Render workspace and example file
    render(target_include_dir, template_vars, 'workspace.h.jinja', 'workspace.h')
    render(target_src_dir, template_vars, 'example.c.jinja', 'example.c')
    render(target_dir, template_vars, 'CMakeLists.txt.jinja', 'CMakeLists.txt')







    # Generate project
    # cwd = os.getcwd()
    # os.chdir(target_dir)
    # call(["cmake", "-DEMBEDDED=%i" % embedded_flag, ".."])
    # os.chdir(cwd)

    #render(target_src_dir, template_vars, 'osqp_cg_data.c.jinja',
    #       'osqp_cg_data.c')
    #render(target_dir, template_vars, 'example_problem.c.jinja',
    #       'example_problem.c')
