# Add the CSC tools sources

set(csc_tools_src
    ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.c)

set(csc_tools_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.h)

# Add more files that should only be in non-embedded code
if(NOT DEFINED EMBEDDED)
    list(APPEND csc_tools_src
      ${CMAKE_CURRENT_SOURCE_DIR}/csc_utils.c)
    list(APPEND csc_tools_headers
      ${CMAKE_CURRENT_SOURCE_DIR}/csc_utils.h)
endif()

#pass upwards
set(csc_tools ${csc_tools_src} ${csc_tools_headers} PARENT_SCOPE)
set(csc_tools_includes ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
