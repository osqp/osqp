.. _citing :

Citing OSQP
===========

If you use OSQP for published work, we encourage you to put a star on `GitHub <https://github.com/oxfordcontrol/osqp>`_ and cite the accompanying papers:


.. glossary::

    Main paper
        Main algorithm description, derivation and benchmark available in this `paper <https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf>`__.

        .. code:: latex

          @article{osqp,
            author  = {Stellato, Bartolomeo and Banjac, Goran and Goulart, Paul and Bemporad, Alberto and Boyd, Stephen},
            title   = {{{OSQP}}: An Operator Splitting Solver for Quadratic Programs},
            journal = {Mathematical Programming Computation},
            year    = {2020},
            doi     = {10.1007/s12532-020-00179-2},
            url     = {https://doi.org/10.1007/s12532-020-00179-2}
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

    Code generation
        Code generation functionality and example in this `paper <https://stanford.edu/~boyd/papers/pdf/osqp_embedded.pdf>`_.

        .. code:: latex

          @inproceedings{osqp-codegen,
            author    = {Banjac, G. and Stellato, B. and Moehle, N. and Goulart, P. and Bemporad, A. and Boyd, S.},
            title     = {Embedded code generation using the {OSQP} solver},
            booktitle = {IEEE Conference on Decision and Control (CDC)},
            year      = {2017},
            doi       = {10.1109/CDC.2017.8263928},
            url       = {https://doi.org/10.1109/CDC.2017.8263928},
          }
