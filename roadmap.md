## TODO
-   [ ] Add indirect solver
-   [ ] Check interfaces values for linsys solver
-   [ ] Make sure codegen checks that the solver is SUITESPARSE_LDL
-   [ ] Add functions to update rho and sigma (by updating the KKT matrix entries)
-   [ ] Replace EMBEDDED flag in C with:  EMBEDDED, EMBEDDED_PARAMETERS_VECTORS, EMBEDDED_PARAMETERS_MATRICES (all true/false)
-   [ ] Implement cheaper dual residual computation: (only one matrix-vector computation)
-   [ ] Stress tests Maros Meszaros
-   [x] Link to CVXPY

### Test Problems

-   [QPLIB2014](http://www.lamsade.dauphine.fr/QPlib2014/doku.php)
-   [Maros and Meszaros Convex Quadratic Programming](https://github.com/YimingYAN/QP-Test-Problems) Test Problem Set



### Examples with bad convergence
-   Add cost function and bounds scaling
-   See `interfaces/python/examples/bad_convergence`
