#ifndef TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP
#define TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP

#include <string>
#include <vector>

namespace xla {
class HloModule;
}  // namespace xla

namespace tf_hlo_trace {

struct hlo_module_trace_insturmentation_metadata {
  std::vector<std::string> value_desriptors;
};

void make_source_locations_unique(xla::HloModule* hlo_module);
hlo_module_trace_insturmentation_metadata insert_trace_instrumentation(
    xla::HloModule* hlo_module);

}  // namespace tf_hlo_trace

#endif  // TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP
