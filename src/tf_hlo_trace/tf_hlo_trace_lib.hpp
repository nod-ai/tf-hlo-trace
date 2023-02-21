#ifndef TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP
#define TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP

namespace xla {
class HloModule;
}  // namespace xla

namespace tf_hlo_trace {

struct hlo_module_trace_insturmentation_metadata {};

void make_source_locations_unique(xla::HloModule* hlo_module);
hlo_module_trace_insturmentation_metadata insert_trace_instrumentation(
    xla::HloModule* hlo_module);

}  // namespace tf_hlo_trace

#endif  // TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP
