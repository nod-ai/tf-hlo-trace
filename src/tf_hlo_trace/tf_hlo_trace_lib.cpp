#include <tensorflow/compiler/xla/hlo/ir/hlo_computation.h>
#include <tensorflow/compiler/xla/hlo/ir/hlo_module.h>

#include <tf_hlo_trace/tf_hlo_trace_lib.hpp>

namespace tf_hlo_trace {

void make_source_locations_unique(xla::HloModule* hlo_module) {
  for (int64_t computation_idx = 0;
       computation_idx < hlo_module->computation_count(); ++computation_idx) {
    xla::HloComputation* hlo_computation =
        hlo_module->mutable_computation(computation_idx);
    size_t instruction_idx = 0;
    for (xla::HloInstruction* instruction : hlo_computation->instructions()) {
      xla::OpMetadata op_metadata = instruction->metadata();
      std::string source_file = op_metadata.source_file() + "[" +
                                instruction->name() + ":" +
                                std::to_string(computation_idx) + ":" +
                                std::to_string(instruction_idx) + "]";
      op_metadata.set_source_file(source_file);
      instruction->set_metadata(op_metadata);
      ++instruction_idx;
    }
  }
}

hlo_module_trace_insturmentation_metadata insert_trace_instrumentation(
    xla::HloComputation* hlo_computation) {
  for (xla::HloInstruction* instruction : hlo_computation->instructions()) {
  }
  return hlo_module_trace_insturmentation_metadata();
}

hlo_module_trace_insturmentation_metadata insert_trace_instrumentation(
    xla::HloModule* hlo_module) {
  xla::HloComputation* entry_computation = hlo_module->entry_computation();
  return insert_trace_instrumentation(entry_computation);
}

}  // namespace tf_hlo_trace
