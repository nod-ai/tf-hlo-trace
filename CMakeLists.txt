cmake_minimum_required (VERSION 3.25)
project (tf-hlo-trace)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard version.")
add_compile_options(-Werror=return-type)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_testing()

add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/googletest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/build_tools")
find_package(TensorflowCC REQUIRED)

add_subdirectory(tests)

add_library(tf_hlo_trace_lib
    src/tf_hlo_trace/tf_hlo_trace_lib.cpp
)
target_include_directories(tf_hlo_trace_lib
    PUBLIC src
)
target_link_libraries(tf_hlo_trace_lib PRIVATE TensorflowCC::TensorflowCC)

pybind11_add_module(tf_hlo_trace_py_ext
    src/tf_hlo_trace/tf_hlo_trace_py_ext.cpp
)
target_include_directories(tf_hlo_trace_py_ext
    PUBLIC src
)
target_link_libraries(
    tf_hlo_trace_py_ext
    PRIVATE tf_hlo_trace_lib
    PRIVATE TensorflowCC::TensorflowCC
)

install(
    TARGETS tf_hlo_trace_py_ext
    COMPONENT python_bindings
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
