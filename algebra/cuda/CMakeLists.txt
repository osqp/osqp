file(
  GLOB
  SRC_FILES
  CONFIGURE_DEPENDS
  *.h
  *.cu
  src/*.cu
  lin_sys/indirect/*.h
  lin_sys/indirect/*.cu)

target_sources(OSQPLIB PRIVATE ${SRC_FILES})

target_include_directories(OSQPLIB PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} include lin_sys/indirect)
