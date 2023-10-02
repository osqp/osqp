Profiling
=========

OSQP includes two mechanisms for measuring the amount of time spent in the solver:

* Basic profiling - provides simple measurements of start/stop times for the solver
* Advanced profiling - Annotates regions of the code so external profilers (e.g. NVidia Nsight, Intel VTune, etc.)
  can measure the time of each part of the solver.

Basic Profiling
---------------

Basic solver profiling is enabled by the :code:`OSQP_ENABLE_PROFILING` CMake build option, and uses the system's
clock to time the setup and solve phases.
The runtime for the solver is displayed alongside the solver status when the solver finishes, and is also contained
inside fields of the :cpp:struct:`OSQPInfo` structure.

Advanced Profiling
------------------

Advanced solver profiling is available with the use of the following external tools:

* :ref:`NVidia Nsight<adv_profile_nvidia_nsight>`
* :ref:`Intel VTune<adv_profile_intel_vtune>`

While these profilers will analyze and report on any configuration of OSQP, when OSQP is built with the
:code:`OSQP_PROFILER_ANNOTATIONS` CMake build option, OSQP will report to the profiler when sections
of the code start/end, allowing more fine-grained timing of individual sections of the solver phase.


.. _adv_profile_nvidia_nsight:

NVidia Nsight
^^^^^^^^^^^^^

When using the NVidia CUDA algebra backend with the profiler annotations enabled, only the `NVidia Nsight`_ profiler should
be used.
This requires the use of the NVidia NVTX library, which is selected using the :code:`OSQP_PROFILER_ANNOTATIONS=nvtx` CMake
option during configuration.

Once compiled with the NVTX library support, OSQP will report profiling information when run using the `NVidia Nsight`_
profiler tool. The timing information will be namespaced into the :code:`osqp` domain, and is reported in the NVTX ranges
section of any reports produced.

.. _NVidia Nsight: https://developer.nvidia.com/nsight-graphics


Command line profiling
~~~~~~~~~~~~~~~~~~~~~~

To use the command line to profile the demo executable with the CUDA algebra and report the OSQP timings,
the following should be run on Linux:

.. code:: bash

   nsys profile -o <report_name> osqp_demo
   nsys analyze <report_name>.nsys-rep
   nsys stats --report nvtx_sum <report_name>.sqlite


This will then report all the timing information for the OSQP sections of the code in a table on the command line:

.. code::

   ** NVTX Range Summary (nvtx_sum):

   Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)    Style                    Range                 
   --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  -------  ---------------------------------------
      47.8      337,159,170          1  337,159,170.0  337,159,170.0  337,159,170  337,159,170           0.0  PushPop  osqp:Problem setup                     
      13.1       92,137,833          1   92,137,833.0   92,137,833.0   92,137,833   92,137,833           0.0  PushPop  osqp:Solving optimization problem      
       9.4       66,362,729         94      705,986.5      466,710.5      278,144   13,770,989   1,435,295.6  PushPop  osqp:Solve the linear system           
       9.2       64,993,272         90      722,147.5      540,812.5      352,410   14,033,099   1,448,686.7  PushPop  osqp:ADMM iteration                    
       8.4       59,295,017         90      658,833.5      482,775.0      295,428   13,872,670   1,437,104.2  PushPop  osqp:KKT system solve in ADMM iteration
       5.4       37,863,537          1   37,863,537.0   37,863,537.0   37,863,537   37,863,537           0.0  PushPop  osqp:Problem data scaling              
       2.4       17,114,751          2    8,557,375.5    8,557,375.5      877,863   16,236,888  10,860,470.7  PushPop  osqp:Initialize linear system solver   
       2.0       13,913,196          1   13,913,196.0   13,913,196.0   13,913,196   13,913,196           0.0  PushPop  osqp:Solution polishing                
       1.3        9,058,039        301       30,093.2       27,691.0       23,970       88,999       8,018.7  PushPop  osqp:Matrix-vector multiplication      
       0.8        5,615,895         90       62,398.8       55,515.5       47,176      158,284      19,590.5  PushPop  osqp:Vector updates in ADMM iteration  
       0.2        1,712,233         90       19,024.8       16,265.5       14,119       72,884       7,576.9  PushPop  osqp:Projection in ADMM iteration      

GUI Profiling
~~~~~~~~~~~~~

To use the `NVidia Nsight`_ GUI to profile OSQP, ensure that the `Collect NVTX trace` project option is selected, and the 
:code:`osqp` domain is included in any NVTX domain filters that are setup.
Then, the OSQP sections will be shown on the timeline, and all the OSQP sections and events can be seen in the `Events view`.
A sample of the Nsight GUI is shown below.

.. image:: ../_static/img/NsightPanel.png
    :alt: Sample Nsight timeline and event panel


.. _adv_profile_intel_vtune:

Intel VTune
^^^^^^^^^^^

When using the built-in or MKL algebra backends, OSQP can be profiled using the `Intel VTune`_ profiler.
OSQP reports its information to VTune using the ITT (Instrumentation and Tracing Technology APIs) library,
and this can be selected when building OSQP by using the :code:`OSQP_PROFILER_ANNOTATIONS=itt` CMake
option during configuration.
OSQP reports the various parts of the solver as a `Task` under the ::code:`osqp` domain.

.. _Intel VTune: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html#gs.6g073h


GUI Profiling
~~~~~~~~~~~~~

To use `Intel VTune`_ to profile OSQP with the profiler annotations, run a `Hotspot` analysis, and ensure the
`Analyze user tasksm events and counters` option is selected.
Once run, the OSQP annotations can be seen in the application's trace in the `Platform` timeline view, and also in the tooltip when 
the mouse hovers over an item in the timeline, as shown below.

.. image:: ../_static/img/VTune_Timeline.png
    :alt: Sample VTune timeline panel

More exact information about the various timings and all the function calls in each part of the solver can be viewed
on the `Bottom-up` tab of the analysis window, with the `Task Type / Function / Call Stack` grouping, as shown below.

.. image:: ../_static/img/VTune_TaskList.png
    :alt: Sample VTune GUI task list