# Operator Splitting QP Solver

Python implementation of the operator splitting QP solver for problems in the form
```
minimize        x'Qx + c'x
subject to      Aeq x = beq
                Aineq x <= bineq
                l <= x <= u
```

## TODO

- [ ] Do preconditioning/equilibration
- [ ] JIT Compilation (Numba ?) to speedup results
- [ ] Timer and compare to other QP solvers
- [ ] Presolver
- [ ] Infeasibility detection: e.g. http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7040300
- [ ] Unboundedness detection
- [ ] Stepsize selection
- [ ] Warm starting
- [ ] Polishing
