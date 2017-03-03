# Minimum version required
cmake_minimum_required (VERSION 3.5)

# Project name
project (osqp)

# project version --> Does it make sense to have it here?
# set(VERSION_MAJOR 0)
# set(VERSION_MINOR 0)
# set(VERSION_PATCH 0)

# Set the output folder where your program will be created
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)


# Detect operating system
# ----------------------------------------------
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    add_definitions(-DIS_LINUX)
endif()
if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    add_definitions(-DIS_MAC)
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_definitions(-DIS_WINDOWS)
endif()



# Set options
# ----------------------------------------------
# Is the code generated for embedded platforms?
#   1 :   Yes. Matrix update not allowed.
#   2 :   Yes. Matrix update allowed.

if (NOT DEFINED EMBEDDED)
    message(STATUS "Embedded is OFF")
else()
    message(STATUS "Embedded is ${EMBEDDED}")
    message(STATUS "Passing EMBEDDED flag to compiler")
    add_definitions(-DEMBEDDED=${EMBEDDED})
endif()

# Is printing enabled?
option (PRINTING "Enable solver printing" ON)
if (DEFINED EMBEDDED)
    message(STATUS "Disabling printing for embedded")
    set(PRINTING OFF)
endif()
message(STATUS "Printing is ${PRINTING}")
if (PRINTING)
    add_definitions(-DPRINTING)
endif (PRINTING)

# Is profiling enabled?
option (PROFILING "Enable solver profiling (timing)" ON)
if (DEFINED EMBEDDED)
    message(STATUS "Disabling profiling for embedded")
    set(PROFILING OFF)
endif()
message(STATUS "Profiling is ${PROFILING}")
if (PROFILING)
    add_definitions(-DPROFILING)
endif (PROFILING)

# Use floats instead of integers
option (DFLOAT "Use float numbers instead of doubles" OFF)
message(STATUS "Floats are ${DFLOAT}")
if (DFLOAT)
    add_definitions(-DDFLOAT)
endif (DFLOAT)

# Use long integers for indexing
option (DLONG "Use long integers for indexing" ON)
message(STATUS "Long integers are ${DLONG}")
if (DLONG)
    add_definitions(-DDLONG)
endif (DLONG)


# Set Compiler flags
# ----------------------------------------------
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm")      # Include math
# Include real time library in linux
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lrt")
endif()


# Include header directory
# ----------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# Set sources
# ----------------------------------------------
if(NOT DEFINED EMBEDDED)
    set(
        osqp_src
        src/auxil.c
        src/cs.c
        src/kkt.c
        src/lin_alg.c
        src/osqp.c
        src/polish.c
        src/proj.c
        src/scaling.c
        src/util.c
    )

    set(
        osqp_headers
        include/auxil.h
        include/constants.h
        include/cs.h
        include/glob_opts.h
        include/kkt.h
        include/lin_alg.h
        include/lin_sys.h
        include/osqp.h
        include/types.h
        include/polish.h
        include/proj.h
        include/scaling.h
        include/util.h
    )
else()
    # If embedded is enabled, do not use cs, kkt, polish
    set(
        osqp_src
        src/auxil.c
        src/lin_alg.c
        src/osqp.c
        src/kkt.c
        src/proj.c
        src/scaling.c
        src/util.c
    )

    set(
        osqp_headers
        include/auxil.h
        include/constants.h
        include/glob_opts.h
        include/lin_alg.h
        include/lin_sys.h
        include/kkt.h
        include/osqp.h
        include/types.h
        include/proj.h
        include/scaling.h
        include/util.h
    )
endif()

# Create  Library
# ----------------------------------------------

# Add linear systems solver
add_subdirectory(lin_sys)

# Create osqp shared library
add_library (osqpdir SHARED ${osqp_src} ${osqp_headers} $<TARGET_OBJECTS:linsys_direct>)

add_library (osqpdirstatic STATIC ${osqp_src} ${osqp_headers} $<TARGET_OBJECTS:linsys_direct>)

# Create demo executable
add_executable (osqp_demo_direct ${PROJECT_SOURCE_DIR}/examples/osqp_demo_direct.c)
target_link_libraries (osqp_demo_direct osqpdir)



# Add testing
# ----------------------------------------------
# Direct linear solver testing
include_directories(tests)
add_executable(osqp_tester_direct ${PROJECT_SOURCE_DIR}/tests/osqp_tester_direct.c)
target_link_libraries (osqp_tester_direct osqpdir)

# Add testing
include(CTest)
enable_testing()
add_test(NAME direct COMMAND $<TARGET_FILE:osqp_tester_direct>)
