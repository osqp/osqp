.. _citing :

Citing OSQP
===========

If you use OSQP for published work, we encourage you to cite the accompanying papers:


.. glossary::

    Main paper
        Main algorithm description, derivation and benchmarks. (Coming soon!).

    Infeasibility detection
        Infeasibility detection proofs using ADMM (also for general conic programs) in this `paper <http://www.optimization-online.org/DB_FILE/2017/06/6058.pdf>`_.

        .. code:: latex

          @article{osqp-infeasibility,
            title   = {Infeasibility detection in the alternating direction method of multipliers for convex optimization},
            author  = {Goran Banjac and Paul Goulart and Bartolomeo Stellato and Stephen Boyd},
            journal = {optimization-online.org},
            year    = {2017},
            url     = {http://www.optimization-online.org/DB_HTML/2017/06/6058.html},
          }

    Code generation
        Code generation functionality.

        .. code:: latex

          @inproceedings{osqp-codegen,
            author = {Banjac, G. and Stellato, B. and Moehle, N. and Goulart, P. and Bemporad, A. and Boyd, S.},
            title = {Embedded code generation using the {OSQP} solver},
            booktitle = {{IEEE} Conference on Decision and Control ({CDC})},
            year = {2017},
            month = dec,
          }


