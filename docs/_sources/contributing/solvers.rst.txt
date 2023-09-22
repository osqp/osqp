Interfacing new linear system solvers
=====================================
OSQP is designed to be easily interfaced to new linear system solvers via dynamic library loading.
To add a linear system solver interface you need to edit the :code:`lin_sys/` directory subfolder :code:`direct/` or :code:`indirect/` depending on the type of solver.
Create a subdirectory with your solver name with four files:

* Dynamic library loading: :code:`mysolver_loader.c` and :code:`mysolver_loader.h`.
* Linear system solution: :code:`mysolver.c` and :code:`mysolver.h`.

We suggest you to have a look at the `MKL Pardiso solver interface <https://github.com/osqp/osqp/tree/master/lin_sys/direct/pardiso>`_ for more details.

Dynamic library loading
^^^^^^^^^^^^^^^^^^^^^^^
In this part define the methods to load the shared library and obtain the functions required to solve the linear system.
The main functions to be exported are :code:`lh_load_mysolver(const char* libname)` and :code:`lh_unload_mysolver()`.
In addition, the file :code:`mysolver_loader.c` must define static function pointers to the shared library functions to be loaded.

Linear system solution
^^^^^^^^^^^^^^^^^^^^^^
In this part we define the core of the interface: **linear system solver object**.
The main functions are the external method

* :code:`init_linsys_solver_mysolver`: create the instance and perform the setup

and the internal methods of the object

* :code:`free_linsys_solver_mysolver`: free the instance
* :code:`solve_linsys_mysolver`: solve the linear system
* :code:`update_matrices`: update problem matrices
* :code:`update_rho_vec`: update :math:`\rho` as a diagonal vector.

After the initializations these functions are assigned to the internal pointers so that, for an instance :code:`s` they can be called as :code:`s->free`, :code:`s->solve`, :code:`s->update_matrices` and :code:`s->update_rho_vec`.

The linear system solver object is defined in :code:`mysolver.h` as follows

.. code:: c

        typedef struct mysolver mysolver_solver;

        struct mysolver {
            // Methods
            enum osqp_linsys_solver_type type; // Linear system solver defined in constants.h

            OSQPInt (*solve)(struct mysolver * self, OSQPFloat * b);
            void    (*free)(struct mysolver * self);
            OSQPInt (*update_matrices)(struct mysolver* self, const OSQPCscMatrix* P, const OSQPCscMatrix* A);
            OSQPInt (*update_rho_vec)(struct mysolver* self, const OSQPFloat * rho_vec, OSQPFloat rho_sc);

            // Attributes
            OSQPInt nthreads; // Number of threads used (required!)

            // Internal attributes of the solver
            ...

            // Internal attributes required for matrix updates
            OSQPInt *PtoKKT, *AtoKKT;    ///< Index of elements from P and A to KKT matrix
            OSQPInt *rhotoKKT;            ///< Index of rho places in KKT matrix
            ...

        };

        // Initialize mysolver solver
        OSQPInt init_linsys_solver_mysolver(mysolver_solver** s, const OSQPCscMatrix* P, const OSQPCscMatrix* A, const OSQPFloat * rho_vec, const OSQPSettings *settings, OSQPInt polish);

        // Solve linear system and store result in b
        OSQPInt solve_linsys_mysolver(mysolver_solver* s, OSQPFloat* b, OSQPInt admm_iter);

         // Update linear system solver matrices
        OSQPInt update_linsys_solver_matrices_mysolver(mysolver_solver* s, const OSQPCscMatrix* P, const OSQPCscMatrix* A);

        // Update rho_vec parameter in linear system solver structure
        OSQPInt update_linsys_solver_rho_vec_mysolver(mysolver_solver* s, const OSQPFloat* rho_vec);

        // Free linear system solver
        void free_linsys_solver_mysolver(mysolver_solver* s);


The function details are coded in the :code:`mysolver.c` file.
