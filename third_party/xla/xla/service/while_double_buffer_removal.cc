/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/while_double_buffer_removal.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

struct LoopInfo {
  // Tuple index into the loop's parameter tuple of the induction variable.
  int64_t indvar_index;

  // Loop trip count.
  int64_t trip_count;
};

// To guarantee that the entire shape is written to, all indices must be
// zero except for one, which must be the loop induction variable.
bool MatchDynamicSliceInDim(HloInstruction* ds, const LoopInfo& loop_info) {
  // Check that the DUS is a DynamicUpdateSlice.
  HloInstruction* to_be_sliced;
  if (!Match(ds,
             match::DynamicSlice().WithOperand(0, match::Op(&to_be_sliced)))) {
    std::cout << ds->name() << " here4" << std::endl;
    return false;
  }

  if (!Match(to_be_sliced, match::GetTupleElement())) {
    std::cout << ds->name() << " here5" << std::endl;
    return false;
  }

  int64_t ds_dim = -1;
  for (int64_t operand_index = 1; operand_index < ds->operand_count();
       ++operand_index) {
    HloInstruction* operand = ds->mutable_operand(operand_index);
    // All constants must be zero in order to write the entire shape.
    if (Match(operand, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (offset.value() != 0) {
        ds_dim = -1;
        break;
      }
    }

    HloInstruction* slice_offset = operand;
    // Check that the update offset is the loop induction variable.
    if (Match(slice_offset, match::GetTupleElement(match::Parameter(),
                                                   loop_info.indvar_index))) {
      ds_dim = operand_index - 1;
    }
  }

  if (ds_dim == -1) {
    std::cout << ds->name() << " here6" << std::endl;
    return false;
  }

  // The shape's broadcast_dim must be exactly equal to the loop trip count.
  if (to_be_sliced->shape().dimensions(ds_dim) != loop_info.trip_count) {
    std::cout << ds->name() << " here7" << std::endl;
    return false;
  }

  return true;
}

// To guarantee that the entire shape is written to, all indices must be
// zero except for one, which must be the loop induction variable.
bool MatchDynamicUpdateSliceInDim(HloInstruction* dus, HloInstruction* user,
                                  const LoopInfo& loop_info) {
  // Check that the DUS is a DynamicUpdateSlice.
  HloInstruction* to_be_updated;
  if (!Match(dus, match::DynamicUpdateSlice().WithOperand(
                      0, match::Op(&to_be_updated)))) {
    return false;
  }
  if (to_be_updated != user) {
    return false;
  }

  int64_t dus_dim = -1;
  for (int64_t operand_index = 2; operand_index < dus->operand_count();
       ++operand_index) {
    HloInstruction* operand = dus->mutable_operand(operand_index);
    // All constants must be zero in order to write the entire shape.
    if (Match(operand, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (offset.value() != 0) {
        dus_dim = -1;
        break;
      }
    }
    // Check that the update offset is the loop induction variable.
    if (Match(operand, match::GetTupleElement(match::Parameter(),
                                              loop_info.indvar_index))) {
      dus_dim = operand_index - 2;
    }
  }

  if (dus_dim == -1) {
    return false;
  }

  // The shape's broadcast_dim must be exactly equal to the loop trip count.
  if (user->shape().dimensions(dus_dim) != loop_info.trip_count) {
    return false;
  }

  return true;
}

bool LoopIndexIsReadOnly(const HloAliasAnalysis& alias_analysis,
                         HloInstruction* while_instr, int64_t idx) {
  const HloDataflowAnalysis& dataflow_analysis =
      alias_analysis.dataflow_analysis();
  return !(
      dataflow_analysis.GetValueSet(while_instr->while_init(), {idx})
              .values()
              .size() > 1 ||
      dataflow_analysis.GetValueSet(while_instr, {idx}).values().size() > 1 ||
      dataflow_analysis.GetUniqueValueAt(while_instr, {idx}) !=
          dataflow_analysis.GetUniqueValueAt(while_instr->while_init(), {idx}));
}

std::vector<std::pair<HloInstruction*, HloInstruction*>> LoopHasDoubleBuffer(
    const HloAliasAnalysis& alias_analysis, HloInstruction* while_instr,
    const LoopInfo& loop_info) {
  HloComputation* computation = while_instr->while_body();
  HloInstruction* body_param = computation->parameter_instruction(0);

  // Finding the buffer indices
  std::vector<HloInstruction*> possible_buffers;
  for (int64_t param_idx = 0;
       param_idx < while_instr->while_init()->operand_count(); ++param_idx) {
    for (HloInstruction* gte : body_param->users()) {
      if (!Match(gte, match::GetTupleElement().WithTupleIndex(param_idx))) {
        continue;
      }
      if (gte->operand(0) != body_param) {
        continue;
      }
      // The buffer should only be updated.
      if (gte->user_count() > 1) {
        continue;
      }
      for (HloInstruction* gte_user : gte->users()) {
        if (MatchDynamicUpdateSliceInDim(gte_user, gte, loop_info)) {
          // The buffer should be written at the same index
          if (computation->root_instruction()->mutable_operand(param_idx) ==
              gte_user) {
            possible_buffers.push_back(gte);
            std::cout << "buffer index: " << param_idx
                      << ", shape = " << gte->shape().ToString() << gte->name()
                      << ", update_value = "
                      << gte_user->mutable_operand(1)->name() << std::endl;
          }
        }
      }
    }
  }

  // Finding the input indices
  std::vector<HloInstruction*> possible_inputs;
  for (int64_t param_idx = 0;
       param_idx < while_instr->while_init()->operand_count(); ++param_idx) {
    for (HloInstruction* gte : body_param->users()) {
      if (!Match(gte, match::GetTupleElement().WithTupleIndex(param_idx))) {
        continue;
      }
      if (gte->operand(0) != body_param) {
        continue;
      }

      // The input should only be sliced and passed to the next iteration.
      if (gte->user_count() > 2) {
        continue;
      }

      for (HloInstruction* gte_user : gte->users()) {
        std::cout << "checking: " << gte->name() << std::endl;
        if (MatchDynamicSliceInDim(gte_user, loop_info)) {
          // The input should be read-only
          if (LoopIndexIsReadOnly(alias_analysis, while_instr,
                                  gte->tuple_index())) {
            possible_inputs.push_back(gte);
            std::cout << "input" << " index: " << param_idx
                      << ", shape = " << gte->shape().ToString() << gte->name()
                      << std::endl;
          }
        }
      }
    }
  }

  std::vector<std::pair<HloInstruction*, HloInstruction*>> out;
  std::vector<HloInstruction*> unique_inputs;
  for (HloInstruction* buffer : possible_buffers) {
    for (HloInstruction* input : possible_inputs) {
      if (ShapeUtil::Equal(input->shape(), buffer->shape())) {
        // Make sure all the inputs are unique, if we encounter a used input, we
        // move over to the next candidate.
        if (absl::c_find(unique_inputs, input) != unique_inputs.end()) {
          continue;
        }
        unique_inputs.push_back(input);
        out.emplace_back(buffer, input);
        std::cout << buffer->GetModule()->unique_id()
                  << " matching: " << buffer->name() << " with "
                  << input->name() << std::endl;
        break;
      }
    }
  }
  return out;
}

absl::StatusOr<bool> RemoveDoubleBuffers(HloModule* module, int64_t id) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));
  // if (id != module->unique_id()) {
  //   return false;
  // }
  std::cout << "Removing double buffer for " << id << std::endl;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->computations()) {
    absl::c_copy_if(
        comp->instructions(), std::back_inserter(while_instrs),
        [&](const HloInstruction* instr) {
          return instr->opcode() == HloOpcode::kWhile &&
                 // We only remove double buffers in nested loops for now.
                 instr->parent()->IsWhileBodyComputation();
        });
  }
  bool changed = false;
  for (HloInstruction* while_instr : while_instrs) {
    bool removed = false;
    std::optional<int64_t> indvar_index =
        GetLoopInductionVarTupleIdx(while_instr);
    std::optional<int64_t> trip_count =
        ComputeWhileLoopTripCount(while_instr, /*max_brute_force_iters=*/0);
    if (indvar_index.has_value() && trip_count.has_value()) {
      std::cout << "loop: " << while_instr->name() << " -> "
                << trip_count.value() << std::endl;
      LoopInfo loop_info{*indvar_index, *trip_count};
      auto out = LoopHasDoubleBuffer(*alias_analysis, while_instr, loop_info);
      for (const auto& [buffer, input] : out) {
        // We only consider buffers that are allocated inside the loop.
        // Therefore, we skip buffers that are passed as the loop input.
        if (Match(while_instr->while_init()->mutable_operand(
                      buffer->tuple_index()),
                  match::GetTupleElement(match::Parameter()))) {
          continue;
        }
        std::cout << while_instr->name() << " -> "
                  << "<buffer: " << buffer->name() << ", "
                  << "input: " << input->name() << ">" << std::endl;
        TF_RETURN_IF_ERROR(input->ReplaceAllUsesWith(buffer));
        TF_RETURN_IF_ERROR(while_instr->while_init()->ReplaceOperandWith(
            buffer->tuple_index(),
            while_instr->while_init()->mutable_operand(input->tuple_index())));
        if (input->user_count() == 0) {
          TF_RETURN_IF_ERROR(
              while_instr->while_body()->RemoveInstruction(input));
          removed = true;
        }
      }
      if (removed) {
        TF_RETURN_IF_ERROR(TryRemoveDeadWhileParams(while_instr).status());
        changed = true;
      }
      std::cout << "======================================" << std::endl;
    }
  }
  if (changed) {
    TF_RETURN_IF_ERROR(TupleSimplifier{}.Run(module).status());
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> WhileDoubleBufferRemoval::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before WhileDoubleBufferRemoval:";
  XLA_VLOG_LINES(2, module->ToString());

  // TODO: we might want to simplify compare instructions before this. It helps
  // us identify more inputs and buffer
  TF_ASSIGN_OR_RETURN(bool changed, RemoveDoubleBuffers(module, module_id_));

  if (changed) {
    VLOG(2) << "HLO module after WhileDoubleBufferRemoval:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileDoubleBufferRemoval";
  }

  return changed;
}

}  // namespace xla
