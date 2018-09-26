# The Operator Splitting QP Solver
[**Join our forum**](https://groups.google.com/forum/#!forum/osqp) for any questions related to the solver!

**The documentation** is available at [**osqp.org**](https://osqp.org/)

The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        0.5 x' P x + q' x

subject to      l <= A x <= u
```

where `x in R^n` is the optimization variable. The objective function is defined by a positive semidefinite matrix `P in S^n_+` and vector `q in R^n`. The linear constraints are defined by matrix `A in R^{m x n}` and vectors `l in R^m U {-inf}^m`, `u in R^m U {+inf}^m`.


The latest version is `0.4.1`.

<br>

<table>
  <tr>
    <th>System</th>
    <th>Status</th>
    <th>Coverage</th>
  </tr>
  <tr>
    <td>Linux / OSX</td>
    <td><a href="https://travis-ci.org/oxfordcontrol/osqp"><img src="https://travis-ci.org/oxfordcontrol/osqp.svg?branch=master"></a></td>
    <td rowspan="2"><a href="https://coveralls.io/github/oxfordcontrol/osqp?branch=master"><img src="https://coveralls.io/repos/github/oxfordcontrol/osqp/badge.svg?branch=master"></a></td>
  </tr>
  <tr>
    <td>Windows</td>
    <td><a href="https://ci.appveyor.com/project/bstellato/osqp/"><img src="https://ci.appveyor.com/api/projects/status/ik6ct0203pq5esxh/branch/master?svg=true"></a></td>
  </tr>
</table>

<br>

## Citing OSQP

If you are using OSQP for your work, we encourage you to

* [Cite the related papers](https://osqp.org/citing/),
* Put a star on this repository.

**We are looking forward to hearing your success stories with OSQP!** Please [share them with us](mailto:bartolomeo.stellato@gmail.com).


## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/oxfordcontrol/osqp/issues). All types of issues are welcome including bug reports, documentation typos, feature requests and so on.


## Numerical benchmarks
Numerical benchmarks against other solvers are available [here](https://github.com/oxfordcontrol/osqp_benchmarks).

