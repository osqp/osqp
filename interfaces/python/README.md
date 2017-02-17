# Operator Splitting QP Solver: Python Interface

Matlab interface for the OSQP (Operator Splitting Quadratic Program) solver.

The current version is `0.0.0`.

OSQP is a numerical software for solving quadratic programs (QPs) of the form

```
minimize     (1/2) x' P x + q' x
subject to   l <= A x <= u
```


## Installation (still need to distribute the package)
The latest version of OSQP can be installed via `pip`
```python
pip install osqp
```

The interface requires *numpy* and *scipy* to handle vectors and sparse matrices.

### Building from sources
If you want to build this Python extension from source, simply run
```python
python setup.py install
```
after having cloned the repository and the submodule with `git clone --recursive -j8` followed by the address.

## Usage
#### Import
The OSQP module can be imported with:
```python
import osqp
```

The solver is initialized by creating the object:
```python
m = osqp.OSQP()
```

#### Setup
The solver can be setup by running:
```python
m.setup(P=P, q=q, A=A, l=l, u=u, **settings)
```
The arguments `q`, `l`, `u` are Numpy arrays. The elements of `l` and `u` can be +/-infinity. The arguments `P` and `A` are scipy sparse matrices in CSC format. If they are in another sparse format, the interface will attempt to convert them. There is no need to specify all the arguments.

The keyword arguments `**settings` specify the solver settings as follows

| Argument            | Description                         | Default value  |
| ---------------     |-------------------------------------| :--------------|
| `scaling`           | Perform data scaling                |   True         |
| `rho`               | ADMM rho step                       |   1.6          |
| `sigma`             | ADMM sigma step                     |   0.1          |
| `max_iter` *        | Maximum number of iterations        |   2500         |
| `eps_abs`  *        | Absolute tolerance                  |   1e-05        |
| `eps_rel`  *        | Relative tolerance                  |   1e-05        |
| `eps_inf`  *        | Infeasibility tolerance             |   1e-06        |
| `eps_unb`  *        | Unboundedness tolerance             |   1e-06        |
| `alpha`    *        | ADMM overrelaxation parameter       |   1.6          |
| `delta`    *        | Polishing regularization parameter  |   1e-07        |
| `polishing`*        | Perform polishing                   |   True         |
| `verbose`  *        | Print output                        |   True         |
| `warm_start` *      | Perform warm starting               |   True         |
| `scaling_norm`      | Scaling norm                        |   2            |
| `scaling_iter`      | Scaling iterations                  |   3            |
| `pol_refine_iter` * | Refinement iterations in polishing  |   5            |

The settings marked with * can be changed without running the setup method again. See section **Update Settings**.

#### Solve
The problem can be solved by

```python
results = m.solve()
```

The results object contains the primal solution `x`, the dual solution `y` and the `info` object containing the solver statistics defined in the following table


| Member          | Description                         |
| --------------- |-------------------------------------|
| `iter`          | Number of iterations                |
| `status`        | Solver status                       |
| `status_val`    | Solver status code                  |
| `status_polish` | Polishing status                    |
| `obj_val`       | Objective value                     |
| `pri_res`       | Primal residual                     |
| `dua_res`       | Dual residual                       |
| `run_time`      | Total run time                      |
| `setup_time`    | Setup time                          |
| `solve_time`    | Solve time                          |
| `polish_time`   | Polish time                         |


#### Update Problem Vectors
The Python interface allows the user to update the problem vectors `q`, `l`, `u` with new values `q_new`, `l_new`, `u_new` without requiring a new stup and matrix factorization by just running
```python
m.update(q=q_new, l=l_new, u=u_new)
```
The user does not have to specify all the keyword arguments at the same time.

#### Update Settings
Part of the settings can be changed without requiring a new setup. They can be changed by running
```python
m.update_settings(**kwargs)
```
where `kwargs` are the allowed settings that can be updated marked with an asterisk in the **Setup** section.


#### Warm Start variables
Primal and dual variables can be warm-started by running
```python
m.warm_start(x=x_0_new, y=y_0_new)
```
where `x_0_new` and `y_0_new` are the new primal and dual variables


## Run Tests
In order to run the tests with `nosetests` you need to have [**mathprogbasepy**](https://github.com/bstellato/mathprogbasepy) installed.

## TODO
-   [ ]  Add all input data checks
-   [ ]  Extend:
    -   [ ] Edit settings
-   [ ]  Add further unittesting
