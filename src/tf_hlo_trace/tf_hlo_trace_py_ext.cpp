

#include <pybind11/pybind11.h>
#include <tensorflow/compiler/xla/service/hlo_module.h>

#include <tf_hlo_trace/tf_hlo_trace_lib.hpp>

namespace tf_hlo_trace {

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;

PYBIND11_MODULE(tf_hlo_trace_py_ext, m) {
  m.def("make_source_locations_unique", [](xla::HloModule& hlo_module) {
    make_source_locations_unique(&hlo_module);
  });

  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

  m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

}  // namespace tf_hlo_trace
