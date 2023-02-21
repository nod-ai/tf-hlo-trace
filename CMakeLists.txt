cmake_minimum_required (VERSION 3.25)
project (tf-hlo-tracer)

find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/tensorflow_package_path.py"
    OUTPUT_VARIABLE TESORFLOW_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

add_subdirectory(third_party/pybind11)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Werror=return-type)

add_library(tf_hlo_trace_lib
    src/tf_hlo_trace/tf_hlo_trace_lib.cpp
)
target_include_directories(tf_hlo_trace_lib
    PUBLIC src
    PRIVATE "${TESORFLOW_PATH}/include"
)

pybind11_add_module(tf_hlo_trace_py_ext
    src/tf_hlo_trace/tf_hlo_trace_py_ext.cpp
)
target_include_directories(tf_hlo_trace_py_ext
    PUBLIC src
    PRIVATE "${TESORFLOW_PATH}/include"
)
target_link_libraries(tf_hlo_trace_py_ext PRIVATE tf_hlo_trace_lib)

install(
    TARGETS tf_hlo_trace_py_ext
    COMPONENT python_bindings
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .)
