#ifndef TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP
#define TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP

#include <string>
#include <utility>
#include <vector>

namespace xla {
class HloModule;
}  // namespace xla

namespace tf_hlo_trace {

struct hlo_module_trace_insturmentation_metadata {
  hlo_module_trace_insturmentation_metadata(
      size_t result_tuple_original_root_instruction_index,
      std::pair<size_t, size_t> result_tuple_original_instructions_range)
      : result_tuple_original_root_instruction_index(
            result_tuple_original_root_instruction_index),
        result_tuple_original_instructions_range(
            result_tuple_original_instructions_range) {}
  std::vector<std::string> value_desriptors;
  // The index of the original root instruction inside the new root/result
  // tuple instruction.
  size_t result_tuple_original_root_instruction_index;
  // The begin and end indexes in the result tuple of all instruction in the HLO
  // computation.
  std::pair<size_t, size_t> result_tuple_original_instructions_range;
};

void make_source_locations_unique(xla::HloModule* hlo_module);
hlo_module_trace_insturmentation_metadata insert_trace_instrumentation(
    xla::HloModule* hlo_module);

}  // namespace tf_hlo_trace

#endif  // TF_HLO_TRACE_TF_HLO_TRACE_LIB_HPP
