file(
  GLOB
  SRC_FILES
  CONFIGURE_DEPENDS
  *.c
  *.h
  *.cu
  src/*.c
  src/*.cu
  lin_sys/indirect/*.c
  lin_sys/indirect/*.h
  lin_sys/indirect/*.cu)

if(IS_WINDOWS)
  set(HELPER_CUDA_H $ENV{NVCUDASAMPLES_ROOT}/common/inc)
else()
  set(HELPER_CUDA_H ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
endif()

target_sources(OSQPLIB PRIVATE ${SRC_FILES})

target_include_directories(OSQPLIB PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} include lin_sys/indirect ${HELPER_CUDA_H})
