Python Wrappers for OSQP
========================

The Python wrappers for OSQP allow you to interact with the core objects in OSQP
in an intuitive way, with easy switching of algebra backends.

We recommend that you use an environment manager like ``conda`` to create a dedicated
Python environment for the purpose of building the Python wrappers for OSQP.

While building the wrappers, we will be using the following command periodically to
check what algebra backends we have successfully built.

.. code-block:: bash

        $ python -c "from osqp import algebras_available; print(algebras_available())"

See :ref:`Building alternative algebra backends <python_build_algebras>` on how to build and test
different OSQP algebra backends.

Building wrappers against an experimental branch of OSQP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``CMakeLists.txt`` in the root folder of the ``osqp-python`` repository lists the ``osqp`` branch that
is fetched and compiled as part of the build process. The relevant section of this file looks like (at the time
of this writing):

.. code-block::

	FetchContent_Declare(
	  osqp
	  GIT_REPOSITORY https://github.com/osqp/osqp.git
	  GIT_TAG v1.0.0.beta1)

If you wish to build the wrappers against a particular branch or commit of ``osqp`` (for example when adding
wrappers for experimental features not available in the main/master branch of ``osqp``), you can modify the
``GIT_TAG`` property to point to the relevant commit/branch, and run the steps in this document to build the
wrappers against the algebra of your choice.

