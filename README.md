# Operator Splitting QP Solver

OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        1/2*x'Px + q'x
subject to      l <= A x <= u
```

The current version is `0.0.0`.

## TODO (Code)

-   [ ] Implement cheaper dual residual computation: (only one matrix-vector computation)

```
dual_res = || rho * A' * ((alpha - 1) * z_tilde_k - z_k + (2 - alpha) * z_prev) ||_2
```
-   [ ] Check infeasibility in the code with new conditions
-   [ ] Parameter Selection: `rho`, `sigma` and `alpha` from examples (data driven)
-   [ ] Stress tests Maros Meszaros
-   [ ] Link to CVXPY


## TODO (Paper)
-   [ ] Write infeasibility/unbounedness conditions
-   [ ] Write compact algorithm (basic ADMM steps)
-   [ ] Proove convergence to vectors satisfying Farkas lemma
-   [ ] Write examples in the paper


### Other Test Problems

-   [QPLIB2014](http://www.lamsade.dauphine.fr/QPlib2014/doku.php)
-   [Maros and Meszaros Convex Quadratic Programming](https://github.com/YimingYAN/QP-Test-Problems) Test Problem Set
