Changes since last commit
-------------------------
* Created new object-oriented structure for linear system solvers
* Added different norms for equilibration (1/2/inf)

Version 0.1.2 (20 July 2017)
------------------------------
* Added option to terminate with scaled or unscaled residual
* Now Matlab interface does support logical entries for the settings
* Fixed bug in index ordering of sparse matrices of Python interface
* Changed 2-norms to inf-norms
* Fixed code generation bug when scaling is disabled [#7](https://github.com/oxfordcontrol/osqp/issues/7)
* Removed warnings in code-generation for standard <= C99
* Fixed MATLAB 2015b compatibility [#6](https://github.com/oxfordcontrol/osqp/issues/6)


Version 0.1.1 (11 April 2017)
-----------------------------
* Fixed crashes during polishing when factorization fails
* Added package to Pypi
* Fixed relative paths Matlab


Version 0.1.0 (10 April 2017)
-----------------------------
* Linux, Mac and Windows
* Interface to Python 2.7, 3.5+
* Interface to Matlab 2015b+
* Embedded code generation
