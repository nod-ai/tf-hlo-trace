#include <tensorflow/compiler/xla/service/hlo_computation.h>
#include <tensorflow/compiler/xla/service/hlo_module.h>

#include <tf_hlo_trace/tf_hlo_trace_lib.hpp>

namespace tf_hlo_trace {

void make_source_locations_unique(xla::HloModule* hlo_module) {}

hlo_module_trace_insturmentation_metadata insert_trace_instrumentation(
    xla::HloModule* hlo_module) {
  hlo_module->entry_computation();
  return hlo_module_trace_insturmentation_metadata();
}

}  // namespace tf_hlo_trace
