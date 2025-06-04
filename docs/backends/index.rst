Algebra Backends
================

.. toctree::
   :hidden:
   :maxdepth: 0

   python.rst

:code:`OSQP` supports several *algebra backends* that can be used to create and solve QP problems. Currently, these
are :code:`builtin`, :code:`MKL`, and :code:`CUDA`. The :code:`builtin` backend is the default, and always available,
regardless of whether you're using the C interface directly, or any of the language bindings.

The solver types :code:`osqp_linsys_solver_type` that are available to you depend on the backend that you're using.
The following table lists the solver types available for each backend:

+---------------------+----------------+----------------+
| Backend/Solver Type | Direct Solver  | Indirect Solver|
+=====================+================+================+
| builtin             | ✔              | ✘              |
+---------------------+----------------+----------------+
| MKL                 | ✔              | ✔              |
+---------------------+----------------+----------------+
| CUDA                | ✘              | ✔              |
+---------------------+----------------+----------------+

Not all backends are available for all language bindings. The following table lists support for various backends
depending on the language wrappers that you're using for OSQP.

The :code:`Codegen` column indicates whether code-generation is supported for a particular language binding.

+----------+---------+-----+-----+----------+
|          | builtin | MKL | GPU | Codegen  |
+==========+=========+=====+=====+==========+
| Python   | ✔       | ✔   | ✔   | ✔        |
+----------+---------+-----+-----+----------+
| Julia    | ✔       | ✔   | ✔   | ✘        |
+----------+---------+-----+-----+----------+
| Matlab   | ✔       | ✘   | ✘   | ✔        |
+----------+---------+-----+-----+----------+

