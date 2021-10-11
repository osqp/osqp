target_sources(OSQPLIB
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/kkt.h
      ${CMAKE_CURRENT_SOURCE_DIR}/kkt.c
)

target_include_directories(OSQPLIB PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR})
