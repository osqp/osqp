# The Operator Splitting QP Solver
[**Join our forum**](https://groups.google.com/forum/#!forum/osqp) for any questions related to the solver!

**The documentation** is available at [**osqp.readthedocs.io**](http://osqp.readthedocs.io/)

The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        0.5 x' P x + q' x

subject to      l <= A x <= u
```

where `x in R^n` is the optimization variable. The objective function is defined by a positive semidefinite matrix `P in S^n_+` and vector `q in R^n`. The linear constraints are defined by matrix `A in R^{m x n}` and vectors `l in R^m U {-inf}^m`, `u in R^m U {+inf}^m`.


The latest version is `0.1.3`.

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



## Credits

The following people have been involved in the development of OSQP:
-  [Bartolomeo Stellato](https://bstellato.github.io/) (University of Oxford): main development
-  [Goran Banjac](http://users.ox.ac.uk/~sedm4978/) (University of Oxford): main development
-  [Nicholas Moehle](http://web.stanford.edu/~moehle/) (Stanford University): methods, maths, and code generation
-  [Paul Goulart](http://users.ox.ac.uk/~engs1373/) (University of Oxford): methods, maths, and Matlab interface
-  [Alberto Bemporad](http://cse.lab.imtlucca.it/~bemporad/) (IMT Lucca): methods and maths
-  [Stephen Boyd](http://web.stanford.edu/~boyd/) (Stanford University): methods and maths


## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/oxfordcontrol/osqp/issues). All types of issues are welcome including bug reports, documentation typos, feature requests and so on.
