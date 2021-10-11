#===============================================================================
# Copyright 2021 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

set(PACKAGE_VERSION "2021.4.0")

if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()

  if("2021.4.0" MATCHES "^([0-9]+)\\.")
    set(CVF_VERSION_MAJOR "${CMAKE_MATCH_1}")
  else()
    set(CVF_VERSION_MAJOR "2021.4.0")
  endif()

  if(PACKAGE_FIND_VERSION_MAJOR STREQUAL CVF_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  else()
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  endif()

  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
      set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()



if("FALSE")
  return()
endif()


if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "" OR "" STREQUAL "")
  return()
endif()


if(NOT CMAKE_SIZEOF_VOID_P STREQUAL "")
  math(EXPR installedBits " * 8")
  set(PACKAGE_VERSION "${PACKAGE_VERSION} (${installedBits}bit)")
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()
