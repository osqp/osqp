# Operator Splitting QP Solver

Python implementation of the operator splitting QP solver for problems in the form
```
minimize        1/2*x'Px + q'x
subject to      lA <= A x <= uA
```

## TODO

- [x] Stopping criterion
- [x] Do preconditioning/equilibration
- [x] Timer and compare to other QP solvers
- [ ] Presolver
- [ ] Infeasibility detection:
    - Generalize [result](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7040300) to non strongly convex problems
- [ ] Unboundedness detection
- [ ] Stepsize selection: maybe choose
- [x] Warm starting
- [x] Polishing:
    - It works but not very robust yet
    - See [this article](https://arxiv.org/pdf/1609.07478.pdf)


### Other Test Problems

- QPLIB2014 http://www.lamsade.dauphine.fr/QPlib2014/doku.php
- Maros and Meszaros Convex Quadratic Programming Test Problem Set: https://github.com/YimingYAN/QP-Test-Problems
