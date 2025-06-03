import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import osqp
    import numpy as np
    from scipy import sparse
    return np, osqp, sparse


@app.cell
def _(np, osqp, sparse):
    # Define problem data
    P = sparse.csc_matrix([[4, 1], [1, 2]])
    q = np.array([1, 1])
    A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
    l = np.array([1, 0, 0])
    u = np.array([1, 0.7, 0.7])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u)

    # Settings can be changed using .update_settings()
    prob.update_settings(polishing=1)

    # Solve problem
    res = prob.solve()

    # Check solver status
    assert res.info.status == "solved"

    print('Status:', res.info.status)
    print('Objective value:', res.info.obj_val)
    print('Optimal solution x:', res.x)
    print('Done!')
    return


if __name__ == "__main__":
    app.run()
