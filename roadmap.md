## TODO

-   [x] Parameter Selection: `rho`, `sigma` and `alpha` from examples (data driven)
-   [ ] Avoid square roots -> Use eps tests to the 2
-   [ ] Add option in CODEGEN to have floats
-   [x] Add functions to update matrices in Python and Matlab
-   [x] Add functions to update matrices in codegen
-   [x] Replace PySys_WriteStdout by PyErr_SetString(PyExc_ValueError, <message>)
-   [ ] Add functions to update rho and sigma (by updating the KKT matrix entries)
-   [ ] Replace EMBEDDED flag in C with:  EMBEDDED, EMBEDDED_PARAMETERS_VECTORS, EMBEDDED_PARAMETERS_MATRICES (all true/false)
-   [x] Add unittests Python for code generation
-   [x] Add CTRL-C interrupt close function
-   [x] Implement code generation in Matlab
-   [ ] Fix relative criterion for termination condition
-   [ ] Implement cheaper dual residual computation: (only one matrix-vector computation)
-   [ ] Stress tests Maros Meszaros
-   [ ] Link to CVXPY

### Test Problems

-   [QPLIB2014](http://www.lamsade.dauphine.fr/QPlib2014/doku.php)
-   [Maros and Meszaros Convex Quadratic Programming](https://github.com/YimingYAN/QP-Test-Problems) Test Problem Set
