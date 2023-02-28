#include <gtest/gtest.h>
#include <tensorflow/compiler/xla/hlo/ir/hlo_computation.h>
#include <tensorflow/compiler/xla/hlo/ir/hlo_module.h>
#include <tensorflow/compiler/xla/service/hlo_parser.h>

#include <filesystem>
#include <fstream>
#include <streambuf>
#include <string>
#include <tf_hlo_trace/tf_hlo_trace_lib.hpp>
#include <utility>

using namespace tf_hlo_trace;

namespace {

static bool endsWith(std::string_view str, std::string_view suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

std::string data_dir_path = std::string();

std::string read_file(const char* path) {
  std::string res;
  std::ifstream stream(path);
  stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  stream.seekg(0, std::ios::end);
  unsigned long long size = stream.tellg();
  res.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);
  res.assign((std::istreambuf_iterator<char>(stream)),
             std::istreambuf_iterator<char>());
  return res;
}

std::unique_ptr<xla::HloModule> load_hlo_module(const char* path) {
  auto hlo_str = read_file(path);
  tsl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_status =
      xla::ParseAndReturnUnverifiedModule(hlo_str);
  std::unique_ptr<xla::HloModule> hlo_module =
      std::move(hlo_module_status).value();
  return hlo_module;
}

std::unique_ptr<xla::HloModule> load_example_multi_instruction_hlo() {
  auto hlo_path = std::filesystem::path(data_dir_path).parent_path() /
                  "multi_instuction.hlo";
  return load_hlo_module(hlo_path.c_str());
}

TEST(TfHloTrace, MakeSourceLocationsUnique) {
  std::unique_ptr<xla::HloModule> hlo_module =
      load_example_multi_instruction_hlo();
  make_source_locations_unique(&*hlo_module);

  for (int64_t computation_idx = 0;
       computation_idx < hlo_module->computation_count(); ++computation_idx) {
    size_t instruction_idx = 0;
    xla::HloComputation* hlo_computation =
        hlo_module->mutable_computation(computation_idx);
    for (xla::HloInstruction* instruction : hlo_computation->instructions()) {
      xla::OpMetadata op_metadata = instruction->metadata();
      std::string expected_suffux = std::string("[") + instruction->name() +
                                    ":" + std::to_string(computation_idx) +
                                    ":" + std::to_string(instruction_idx) + "]";
      ASSERT_TRUE(endsWith(op_metadata.source_file(), expected_suffux));
      ++instruction_idx;
    }
  }
}

TEST(TfHloTrace, InsertTraceInstrumentation) {
  std::unique_ptr<xla::HloModule> hlo_module =
      load_example_multi_instruction_hlo();
  xla::HloComputation* entry_computation = hlo_module->entry_computation();
  size_t original_instruction_count = entry_computation->instruction_count();
  auto original_instructions_range = entry_computation->instructions();
  xla::HloInstruction* original_root_instruction =
      entry_computation->root_instruction();
  hlo_module_trace_insturmentation_metadata instrumentation_metadata =
      insert_trace_instrumentation(&*hlo_module);
  xla::HloInstruction* root_instruction = entry_computation->root_instruction();
  ASSERT_EQ(root_instruction->opcode(), xla::HloOpcode::kTuple);
  ASSERT_LE(
      instrumentation_metadata.result_tuple_original_root_instruction_index,
      root_instruction->operand_count());
  ASSERT_EQ(
      original_root_instruction,
      root_instruction
          ->operands()[instrumentation_metadata
                           .result_tuple_original_root_instruction_index]);
  ASSERT_EQ(
      original_instruction_count,
      instrumentation_metadata.result_tuple_original_instructions_range.second -
          instrumentation_metadata.result_tuple_original_instructions_range
              .first);
  ASSERT_EQ(
      root_instruction
          ->operands()[instrumentation_metadata
                           .result_tuple_original_instructions_range.first],
      *original_instructions_range.begin());
}

}  // namespace

int main(int argc, char** argv) {
  data_dir_path = std::string(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
