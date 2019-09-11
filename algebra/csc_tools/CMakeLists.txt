# Add the CSC tools sources

target_sources(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csc_math.c
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
