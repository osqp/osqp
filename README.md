# Operator Splitting QP Solver

OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        1/2*x'Px + q'x
subject to      lA <= A x <= uA
```

The current version is `0.0.0`.


## TODO

-   [x] Equilibration
-   [ ] Warm starting: added basic one BUT need to adjust the total run_time
-   [ ] Sublevel API -> Matrix factorization caching

-   [x] Do preconditioning/equilibration
-   [ ] Timer and compare to other QP solvers
-   [ ] Presolver
-   [ ] Infeasibility detection:
    -   Generalize [result](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7040300) to non strongly convex problems
-   [ ] Unboundedness detection
-   [ ] Stepsize selection: maybe choose




### Other Test Problems

- QPLIB2014 http://www.lamsade.dauphine.fr/QPlib2014/doku.php
- Maros and Meszaros Convex Quadratic Programming Test Problem Set: https://github.com/YimingYAN/QP-Test-Problems
