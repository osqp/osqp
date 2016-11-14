# import test_utils.codeutils as cu
from qptests.diesel.load_data import gen_diesel_test
from qptests.chain80w.load_data import gen_chain80w_test

# Generate problem headers for all problems

# Problems list
qp_problems = {"diesel", "chain80w"}


for test in qp_problems:
    exec("gen_%s_test()" % test)
