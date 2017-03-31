# The Operator Splitting QP Solver
[![Build Status](https://travis-ci.org/oxfordcontrol/osqp.svg?branch=master)](https://travis-ci.org/oxfordcontrol/osqp)
[![Coverage Status](https://coveralls.io/repos/github/oxfordcontrol/osqp/badge.svg?branch=master)](https://coveralls.io/github/oxfordcontrol/osqp?branch=master)
[![Gitter](https://badges.gitter.im/oxfordcontrol/osqp.svg)](https://gitter.im/oxfordcontrol/osqp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        0.5 x' P x + q' x

subject to      l <= A x <= u
```

The current version is `0.0.0`.

**Documentation** ( [stable](http://osqp.readthedocs.io/en/stable) | [latest](http://osqp.readthedocs.io/en/latest) )
