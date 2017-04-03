# The Operator Splitting QP Solver
[** Join our forum**](https://groups.google.com/forum/#!forum/osqp) for any questions related to the solver!

**The documentation** is available at [**osqp.readthedocs.io**](http://osqp.readthedocs.io/)

The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        0.5 x' P x + q' x

subject to      l <= A x <= u
```


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
    <td><a href="https://ci.appveyor.com/project/bstellato/osqp/"><img src="https://ci.appveyor.com/api/projects/status/ik6ct0203pq5esxh?svg=true"></a></td>
  </tr>
</table>

<br>
