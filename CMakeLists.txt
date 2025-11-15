cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(JAX_FFI_EXAMPLE_ENABLE_CUDA "Enable CUDA support" OFF)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

message(STATUS "LDFLAGS: $ENV{LDFLAGS}")
message(STATUS "CPPFLAGS: $ENV{CPPFLAGS}")

find_package(OpenMP REQUIRED)

set(CXX_STANDARD 17)


find_package(nanobind CONFIG REQUIRED)


set(filter_dir "src/${SKBUILD_PROJECT_NAME}/filter/csrc")

nanobind_add_module("_filter_cpu" NB_STATIC "${filter_dir}/filter_cpu.cc")
target_include_directories("_filter_cpu" PUBLIC ${XLA_DIR})
target_link_libraries("_filter_cpu" PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS "_filter_cpu" LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/filter)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  nanobind_add_module(_filter_gpu NB_STATIC "${filter_dir}/filter_gpu.cu")
  set_target_properties(_filter_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)

  target_link_libraries(_filter_gpu PRIVATE CUDA::cudart)
  target_include_directories(_filter_gpu PUBLIC ${XLA_DIR})
  install(TARGETS _filter_gpu LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/filter)
else()
  message(STATUS "No CUDA toolkit found, not building GPU module.")
endif()