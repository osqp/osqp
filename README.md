# Operator Splitting QP Solver

OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        1/2*x'Px + q'x
subject to      l <= A x <= u
```

The current version is `0.0.0`.

## TODO

-   [ ] Write infeasibility/unbounedness conditions
-   [ ] Write compact algorithm (basic ADMM steps)
-   [ ] Parameter Selection: `rho`, `sigma` and `alpha` from examples (data driven)
-   [ ] Stress tests Maros Meszaros
-   [ ] Proove convergence to vectors satisfying Farkas lemma
-   [ ] Write examples in the paper
-   [ ] Link to CVXPY


### Other Test Problems

-   [QPLIB2014](http://www.lamsade.dauphine.fr/QPlib2014/doku.php)
-   [Maros and Meszaros Convex Quadratic Programming](https://github.com/YimingYAN/QP-Test-Problems) Test Problem Set
