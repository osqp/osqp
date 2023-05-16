.. _citing :

Citing OSQP
===========

If you use OSQP for published work, we encourage you to put a star on `GitHub <https://github.com/osqp/osqp>`_ and cite the accompanying papers:


.. glossary::

    Main paper
        Main algorithm description, derivation and benchmark available in this `paper <https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf>`__.

        .. code:: latex

          @article{osqp,
            author  = {Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S.},
            title   = {{OSQP}: an operator splitting solver for quadratic programs},
            journal = {Mathematical Programming Computation},
            volume  = {12},
            number  = {4},
            pages   = {637--672},
            year    = {2020},
            doi     = {10.1007/s12532-020-00179-2},
            url     = {https://doi.org/10.1007/s12532-020-00179-2},
          }

    Infeasibility detection
        Infeasibility detection proofs using ADMM (also for general conic programs) in this `paper <https://stanford.edu/~boyd/papers/pdf/admm_infeas.pdf>`__.

        .. code:: latex

          @article{osqp-infeasibility,
            author  = {Banjac, G. and Goulart, P. and Stellato, B. and Boyd, S.},
            title   = {Infeasibility detection in the alternating direction method of multipliers for convex optimization},
            journal = {Journal of Optimization Theory and Applications},
            year    = {2019},
            volume  = {183},
            number  = {2},
            pages   = {490--519},
            doi     = {10.1007/s10957-019-01575-y},
            url     = {https://doi.org/10.1007/s10957-019-01575-y},
          }

    GPU implementation
        GPU implementation and PCG method for solving linear systems in this `paper <https://doi.org/10.1016/j.jpdc.2020.05.021>`__.

        .. code:: latex

          @article{osqp-gpu,
            author  = {Schubiger, M. and Banjac, G. and Lygeros, J.},
            title   = {{GPU} acceleration of {ADMM} for large-scale quadratic programming},
            journal = {Journal of Parallel and Distributed Computing},
            year    = {2020},
            volume  = {144},
            pages   = {55--67},
            doi     = {10.1016/j.jpdc.2020.05.021},
            url     = {https://doi.org/10.1016/j.jpdc.2020.05.021},
          }

    Code generation
        Code generation functionality and example in this `paper <https://stanford.edu/~boyd/papers/pdf/osqp_embedded.pdf>`__.

        .. code:: latex

          @inproceedings{osqp-codegen,
            author    = {Banjac, G. and Stellato, B. and Moehle, N. and Goulart, P. and Bemporad, A. and Boyd, S.},
            title     = {Embedded code generation using the {OSQP} solver},
            booktitle = {IEEE Conference on Decision and Control (CDC)},
            year      = {2017},
            doi       = {10.1109/CDC.2017.8263928},
            url       = {https://doi.org/10.1109/CDC.2017.8263928},
          }

    Mixed-integer optimization
        A branch-and-bound solver for mixed-integer quadratic optimization in this `paper <https://stellato.io/assets/downloads/publications/2018/miosqp_ecc.pdf>`__.

        .. code:: latex

          @inproceedings{miosqp,
            author    = {Stellato, B. and Naik, V. V. and Bemporad, A. and Goulart, P. and Boyd, S.},
            title     = {Embedded mixed-integer quadratic optimization using the {OSQP} solver},
            booktitle = {European Control Conference (ECC)},
            year      = {2018},
            doi       = {10.23919/ECC.2018.8550136},
            url       = {https://doi.org/10.23919/ECC.2018.8550136},
          }
