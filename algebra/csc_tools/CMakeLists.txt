# Add the CSC tools sources

target_sources(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.c
    ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.h)

set(csc_tools_src
        ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.c)

set(csc_tools_headers
        ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.h)

# Add more files that should only be in non-embedded code
if(NOT DEFINED EMBEDDED)
  target_sources(OSQPLIB
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csc_utils.c
      ${CMAKE_CURRENT_SOURCE_DIR}/csc_utils.h)
endif()

target_include_directories(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

#pass upwards
set(csc_tools ${csc_tools_src} ${csc_tools_headers} PARENT_SCOPE)
set(csc_tools_includes ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)