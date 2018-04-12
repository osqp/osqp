# CMake module to find R
# - Try to find R.  If found, defines:
#
#  R_FOUND        - system has R
#  R_EXEC         - the system R command
#  R_ROOT_DIR     - the R root directory
#  R_INCLUDE_DIRS - the R include directories

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_FIND_APPBUNDLE "LAST")
endif()

find_program(R_EXEC NAMES R R.exe)

#---Find includes and libraries if R exists
if(R_EXEC)

  set(R_FOUND TRUE)

  execute_process(WORKING_DIRECTORY .
                  COMMAND ${R_EXEC} RHOME
                  OUTPUT_VARIABLE R_ROOT_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_path(R_INCLUDE_DIRS R.h
            HINTS ${R_ROOT_DIR}
            PATHS /usr/local/lib /usr/local/lib64 /usr/share
            PATH_SUFFIXES include R/include)
endif()

mark_as_advanced(R_FOUND R_EXEC R_ROOT_DIR R_INCLUDE_DIRS)
