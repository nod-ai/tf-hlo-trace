add_executable(tf_hlo_trace_test tf_hlo_trace_test.cpp)
target_link_libraries(
    tf_hlo_trace_test
    PRIVATE tf_hlo_trace_lib
    PRIVATE gtest_main
    PRIVATE TensorflowCC::TensorflowCC
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/multi_instuction.hlo"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/multi_instuction.hlo"
        "${CMAKE_CURRENT_BINARY_DIR}/multi_instuction.hlo"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/multi_instuction.hlo"
)
add_custom_target(
    tf_hlo_trace_test_files
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/multi_instuction.hlo"
)

add_dependencies(tf_hlo_trace_test tf_hlo_trace_test_files)

add_test(
    NAME tf_hlo_trace_test
    COMMAND tf_hlo_trace_test
)
