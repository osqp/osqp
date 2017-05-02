"""
Run all Examples in the OSQP paper
"""

# Import problems
from scripts.portfolio.portfolio_example import run_portfolio_example
from scripts.huber.huber_example import run_huber_example


'''
Portfolio
'''
#run_portfolio_example()

'''
Huber fitting
'''
run_huber_example()
