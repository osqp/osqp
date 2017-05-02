"""
Run all Examples in the OSQP paper
"""

# Import problems
from scripts.portfolio.portfolio_example import run_portfolio_example
from scripts.lasso.lasso_example import run_lasso_example
from scripts.eq_qp.eq_qp_example import run_eq_qp_example
from scripts.huber.huber_example import run_huber_example
from scripts.svm.svm_example import run_svm_example


# Run problems
run_eq_qp_example()
# run_portfolio_example()
# run_lasso_example()
# run_huber_example()
# run_svm_example()
