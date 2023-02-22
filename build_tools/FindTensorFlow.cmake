include(FindPackageHandleStandardArgs)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/tensorflow_package_path.py"
    OUTPUT_VARIABLE TESORFLOW_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
set(TensorFlow_INCLUDE_DIRS "${TESORFLOW_PATH}/include")
if(NOT EXISTS "${TensorFlow_INCLUDE_DIRS}")
    set(TensorFlow_INCLUDE_DIRS "NOTFOUND")
endif()

find_library(
    TensorFlow_LIBRARIES libtensorflow_framework.so.2
    PATHS "${TESORFLOW_PATH}"
    NO_DEFAULT_PATH
    NO_CACHE
)

if(TensorFlow_INCLUDE_DIRS AND TensorFlow_LIBRARIES)
    add_library(TensorFlow::TensorFlow UNKNOWN IMPORTED)
    set_target_properties(
        TensorFlow::TensorFlow
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TensorFlow_INCLUDE_DIRS}"
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${TensorFlow_LIBRARIES}"
    )
endif()
find_package_handle_standard_args(
    TensorFlow
    REQUIRED_VARS TensorFlow_INCLUDE_DIRS TensorFlow_LIBRARIES
)
