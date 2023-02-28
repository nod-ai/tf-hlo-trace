

#include <pybind11/pybind11.h>
#include <tensorflow/compiler/xla/hlo/ir/hlo_module.h>

#include <tf_hlo_trace/tf_hlo_trace_lib.hpp>

namespace tf_hlo_trace {

namespace py = pybind11;

PYBIND11_MODULE(tf_hlo_trace_py_ext, m) {
  // py::class_<xla::HloModule, std::shared_ptr<xla::HloModule>>
  // hlo_module_class(
  //     m, "HloModule");

  m.def("make_source_locations_unique", [](xla::HloModule& hlo_module) {
    make_source_locations_unique(&hlo_module);
  });
}

}  // namespace tf_hlo_trace
