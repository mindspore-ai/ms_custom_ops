/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <utility>
#include <vector>
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"
#include "ops/c_api/utils/check_utils.h"

namespace ms_custom_ops {
enum class MoeDistributeDispatchV3InputIndex : size_t {
  kMoeDistributeDispatchV3XIndex = 0,
  kMoeDistributeDispatchV3ExpertIdsIndex,
  kMoeDistributeDispatchV3EpWorldSizeIndex,
  kMoeDistributeDispatchV3EpRankIdIndex,
  kMoeDistributeDispatchV3MoeExpertNumIndex,
  kMoeDistributeDispatchV3ScalesIndex,
  kMoeDistributeDispatchV3XActiveMaskIndex,
  kMoeDistributeDispatchV3ExpertScalesIndex,
  kMoeDistributeDispatchV3ElasticInfoIndex,
  kMoeDistributeDispatchV3GroupEpIndex,
  kMoeDistributeDispatchV3GroupTpIndex,
  kMoeDistributeDispatchV3TpWorldSizeIndex,
  kMoeDistributeDispatchV3TpRankIdIndex,
  kMoeDistributeDispatchV3ExpertShardTypeIndex,
  kMoeDistributeDispatchV3SharedExpertNumIndex,
  kMoeDistributeDispatchV3SharedExpertRankNumIndex,
  kMoeDistributeDispatchV3QuantModeIndex,
  kMoeDistributeDispatchV3GlobalBsIndex,
  kMoeDistributeDispatchV3ExpertTokenNumsTypeIndex,
  kMoeDistributeDispatchV3CommAlgIndex,
  kMoeDistributeDispatchV3ZeroExpertNumIndex,
  kMoeDistributeDispatchV3CopyExpertNumIndex,
  kMoeDistributeDispatchV3ConstExpertNumIndex,
  kMoeDistributeDispatchV3InputNums,
};

enum class MoeDistributeDispatchV3OutputIndex : size_t {
  kMoeDistributeDispatchV3ExpandXOutputIndex = 0,
  kMoeDistributeDispatchV3DynamicScalesOutputIndex,
  kMoeDistributeDispatchV3AssistInfoForCombineOutputIndex,
  kMoeDistributeDispatchV3ExpertTokenNumsOutputIndex,
  kMoeDistributeDispatchV3EpRecvCountsOutputIndex,
  kMoeDistributeDispatchV3TpRecvCountsOutputIndex,
  kMoeDistributeDispatchV3ExpandScalesOutputIndex,
  kMoeDistributeDispatchV3OutputNums,
};

struct InputParam {
  ShapeVector x_shape_;
  ShapeVector expert_ids_shape_;
  int64_t ep_world_size_{0};
  int64_t ep_rank_id_{0};
  int64_t moe_expert_num_{0};
  int64_t tp_world_size_{0};
  int64_t tp_rank_id_{0};
  int64_t expert_shard_type_{0};
  int64_t shared_expert_num_{0};
  int64_t shared_expert_rank_num_{0};
  int64_t quant_mode_{0};
  int64_t global_bs_{0};
  int64_t expert_token_nums_type_{0};
  int64_t bs_{0};
  int64_t h_{0};
  int64_t k_{0};
};

struct OutputShapes {
  ShapeVector expand_x;
  ShapeVector dynamic_scales;
  ShapeVector assist_info;
  ShapeVector expert_token_nums;
  ShapeVector ep_recv_counts;
  ShapeVector tp_recv_counts;
  ShapeVector expand_scales;
};

void ValidateParamsComm(const int64_t shared_expert_num, const int64_t shared_expert_rank_num,
                        const int64_t expert_token_nums_type, const std::string &op_name) {
  const bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
  const bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
  const bool is_valid_shared = ((shared_expert_num > 0) && ((shared_expert_rank_num / shared_expert_num) > 0) &&
                                ((shared_expert_rank_num % shared_expert_num) == 0));
  MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
    (is_shared_default || is_no_shared || is_valid_shared), op_name,
    std::string("shared_expert_num and shared_expert_rank_num have obvious value situations:") +
      std::string("1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert_num is "
                  "0,shared_expert_rank_num is ") +
      std::string("0; 3. shared_expert_num is (0, shared_expert_rank_num] and ") +
      std::string("shared_expert_rank_num % shared_expert_num = 0, but the current value is  shared_expert_num:") +
      std::to_string(shared_expert_num) + ", shared_expert_rank_num:" + std::to_string(shared_expert_rank_num));

  MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
    ((expert_token_nums_type == 0) || (expert_token_nums_type == 1)), op_name,
    "The expect token nums type should be 0 or 1, but got " + std::to_string(expert_token_nums_type));
}

class OPS_API MoeDistributeDispatchV3CustomOpFuncImpl : public OpFuncImpl {
 public:
  void ValidateInputsSize(const InferInfoPtrList &input_infos, const std::string &op_name) const {
    constexpr uint32_t kRequiredInputNums =
      static_cast<uint32_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3InputNums);
    MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL((input_infos.size() == kRequiredInputNums), op_name,
                                          " the input size should be equal to: " + std::to_string(kRequiredInputNums) +
                                            ", but got " + std::to_string(input_infos.size()));
  }

  template <typename T>
  T GetInputValue(const InferInfoPtrList &input_infos, MoeDistributeDispatchV3InputIndex index) const {
    return input_infos[static_cast<size_t>(index)]->GetScalarValueWithCheck<T>();
  }

  InputParam ExtractInputParams(const InferInfoPtrList &input_infos, const std::string &op_name) const {
    InputParam params;
    params.x_shape_ =
      input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XIndex)]->GetShape();
    params.expert_ids_shape_ =
      input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertIdsIndex)]
        ->GetShape();
    params.ep_world_size_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3EpWorldSizeIndex);
    params.ep_rank_id_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3EpRankIdIndex);
    params.moe_expert_num_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3MoeExpertNumIndex);
    params.tp_world_size_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3TpWorldSizeIndex);
    params.tp_rank_id_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3TpRankIdIndex);
    params.expert_shard_type_ = GetInputValue<int64_t>(
      input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertShardTypeIndex);
    params.shared_expert_num_ = GetInputValue<int64_t>(
      input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3SharedExpertNumIndex);
    params.shared_expert_rank_num_ = GetInputValue<int64_t>(
      input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3SharedExpertRankNumIndex);
    params.quant_mode_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3QuantModeIndex);
    params.global_bs_ =
      GetInputValue<int64_t>(input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3GlobalBsIndex);
    params.expert_token_nums_type_ = GetInputValue<int64_t>(
      input_infos, MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertTokenNumsTypeIndex);

    params.bs_ = params.x_shape_[kDim0];
    params.h_ = params.x_shape_[kDim1];
    params.k_ = params.expert_ids_shape_[kDim1];
    return params;
  }

  void ValidateParams(const InputParam &params, const std::string &op_name) const {
    ValidateParamsComm(params.shared_expert_num_, params.shared_expert_rank_num_, params.expert_token_nums_type_,
                       op_name);
  }

  bool HasDynamicRank(const InferInfoPtrList &input_infos) const {
    return input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XIndex)]
             ->IsDynamicRank() ||
           input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertIdsIndex)]
             ->IsDynamicRank();
  }

  ShapeArray CreateDynamicRanks() const {
    const ShapeVector dynamic_rank_shape{abstract::Shape::kShapeRankAny};
    return {dynamic_rank_shape, dynamic_rank_shape, dynamic_rank_shape, dynamic_rank_shape,
            dynamic_rank_shape, dynamic_rank_shape, dynamic_rank_shape};
  }

  size_t CalculateLocalExpertNum(const InferInfoPtrList &input_infos, const InputParam &params) const {
    const bool shared_front = (params.expert_shard_type_ == 0);
    size_t local_moe_expert_num = 0;

    if (shared_front) {
      if (params.ep_rank_id_ < params.shared_expert_rank_num_) {
        local_moe_expert_num = 1;
      } else {
        local_moe_expert_num = params.moe_expert_num_ / (params.ep_world_size_ - params.shared_expert_rank_num_);
      }

      if (!input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ElasticInfoIndex)]
             ->IsNone()) {
        local_moe_expert_num = std::max(
          local_moe_expert_num,
          static_cast<size_t>(params.moe_expert_num_ / (params.ep_world_size_ - params.shared_expert_rank_num_)));
      }
    }
    return local_moe_expert_num;
  }

  size_t CalcEpRecvCount(const InputParam &params, size_t local_moe_expert_num) const {
    return (params.tp_world_size_ == kDim2) ? params.ep_world_size_ * local_moe_expert_num * params.tp_world_size_
                                            : params.ep_world_size_ * local_moe_expert_num;
  }

  void SetPartiallyDynamicShapes(OutputShapes &outputs, const InputParam &params, size_t local_moe_expert_num,
                                 size_t ep_recv_counts) const {
    outputs.expand_x = {abstract::Shape::kShapeDimAny, params.h_};
    outputs.dynamic_scales = {abstract::Shape::kShapeDimAny};
    outputs.assist_info = {abstract::Shape::kShapeDimAny};
    outputs.expert_token_nums = {static_cast<int64_t>(local_moe_expert_num)};

    if (IsSoc910b()) {
      outputs.ep_recv_counts = {abstract::Shape::kShapeDimAny};
      outputs.expand_scales = {abstract::Shape::kShapeDimAny};
    } else if (IsSoc910_93()) {
      outputs.ep_recv_counts = {static_cast<int64_t>(ep_recv_counts)};
      outputs.expand_scales = kFakeOutShapes;
    }
    outputs.tp_recv_counts = {params.tp_world_size_};
  }

  ShapeArray ConvertToShapeArray(const OutputShapes &outputs) const {
    return {outputs.expand_x,       outputs.dynamic_scales, outputs.assist_info,  outputs.expert_token_nums,
            outputs.ep_recv_counts, outputs.tp_recv_counts, outputs.expand_scales};
  }

  ShapeArray CreatPartiallyDynamicShapes(const InferInfoPtrList &input_infos, const InputParam &params) const {
    OutputShapes outputs;
    const size_t local_moe_expert_num = CalculateLocalExpertNum(input_infos, params);
    const size_t ep_recv_counts = CalcEpRecvCount(params, local_moe_expert_num);
    SetPartiallyDynamicShapes(outputs, params, local_moe_expert_num, ep_recv_counts);
    return ConvertToShapeArray(outputs);
  }

  std::pair<size_t, size_t> CalculateLocalExpertAndBufferSize(const InferInfoPtrList &input_infos,
                                                              const InputParam &params) const {
    const bool shared_front = (params.expert_shard_type_ == 0);
    const bool is_shared_default = ((params.shared_expert_num_ == 1) && (params.shared_expert_rank_num_ == 0));
    const bool is_no_shared = ((params.shared_expert_num_ == 0) && (params.shared_expert_rank_num_ == 0));
    size_t local_moe_expert_num = 0;
    size_t a = 0;
    const size_t global_bs_real = (params.global_bs_ == 0) ? (params.bs_ * params.ep_world_size_) : params.global_bs_;
    if (shared_front) {
      if (params.ep_rank_id_ < params.shared_expert_rank_num_) {
        local_moe_expert_num = 1;
        const size_t max_bs = global_bs_real / params.ep_world_size_;
        const size_t rank_num_per_shared_expert = params.shared_expert_rank_num_ / params.shared_expert_num_;
        const size_t max_shared_group_num =
          (params.ep_world_size_ + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
        a = max_bs * max_shared_group_num;
      } else {
        local_moe_expert_num = params.moe_expert_num_ / (params.ep_world_size_ - params.shared_expert_rank_num_);
        a = global_bs_real * std::min(local_moe_expert_num, static_cast<size_t>(params.k_));
      }

      if (!input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ElasticInfoIndex)]
             ->IsNone()) {
        if (is_shared_default || is_no_shared) {
          local_moe_expert_num = std::max(
            local_moe_expert_num,
            static_cast<size_t>(params.moe_expert_num_ / (params.ep_world_size_ - params.shared_expert_rank_num_)));
          a = global_bs_real * std::min(local_moe_expert_num, static_cast<size_t>(params.k_));
        } else {
          const size_t max_bs = global_bs_real / params.ep_world_size_;
          const size_t rank_num_per_shared_expert = params.shared_expert_rank_num_ / params.shared_expert_num_;
          const size_t max_shared_group_num =
            (params.ep_world_size_ + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
          a = std::max(
            max_bs * max_shared_group_num,
            static_cast<size_t>(global_bs_real *
                                std::min(static_cast<size_t>(params.moe_expert_num_ /
                                                             (params.ep_world_size_ - params.shared_expert_rank_num_)),
                                         static_cast<size_t>(params.k_))));
          local_moe_expert_num = std::max(
            local_moe_expert_num,
            static_cast<size_t>(params.moe_expert_num_ / (params.ep_world_size_ - params.shared_expert_rank_num_)));
        }
      }
    }
    return {local_moe_expert_num, a};
  }

  void SetPreciseOutputShapes(OutputShapes &outputs, const InputParam &params, size_t local_moe_expert_num,
                              size_t ep_recv_cnt_nums, size_t a) const {
    if (params.tp_world_size_ == kDim0) {
      outputs.expand_x = {static_cast<int64_t>(a), params.h_};
      outputs.dynamic_scales = {static_cast<int64_t>(a)};
    } else {
      outputs.expand_x = {static_cast<int64_t>(a * params.tp_world_size_), params.h_};
      outputs.dynamic_scales = {static_cast<int64_t>(a * params.tp_world_size_)};
    }

    outputs.assist_info = {
      static_cast<int64_t>(std::max(static_cast<int64_t>(params.bs_ * params.k_), static_cast<int64_t>(a * 128)))};
    outputs.expert_token_nums = {static_cast<int64_t>(local_moe_expert_num)};
    outputs.ep_recv_counts = {static_cast<int64_t>(ep_recv_cnt_nums)};
    outputs.tp_recv_counts = {params.tp_world_size_};
    outputs.expand_scales = {static_cast<int64_t>(a)};
  }

  ShapeArray CalcPreciseShapes(const InferInfoPtrList &input_infos, const InputParam &params) const {
    OutputShapes outputs;
    const auto [local_moe_expert_num, a] = CalculateLocalExpertAndBufferSize(input_infos, params);
    size_t ep_recv_cnt_nums = CalcEpRecvCount(params, local_moe_expert_num);
    if (!input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertScalesIndex)]
           ->IsNone()) {
      const size_t global_bs_real = (params.global_bs_ == 0) ? (params.bs_ * params.ep_world_size_) : params.global_bs_;
      ep_recv_cnt_nums =
        params.ep_world_size_ * local_moe_expert_num + 2 * global_bs_real * params.k_ * (params.ep_world_size_ / 8);
    }
    SetPreciseOutputShapes(outputs, params, local_moe_expert_num, ep_recv_cnt_nums, a);
    return ConvertToShapeArray(outputs);
  }

  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();

    ValidateInputsSize(input_infos, op_name);
    InputParam params = ExtractInputParams(input_infos, op_name);
    ValidateParams(params, op_name);
    if (HasDynamicRank(input_infos)) {
      return CreateDynamicRanks();
    }

    if ((params.bs_ == abstract::Shape::kShapeDimAny) || (params.k_ == abstract::Shape::kShapeDimAny)) {
      return CreatPartiallyDynamicShapes(input_infos, params);
    }

    return CalcPreciseShapes(input_infos, params);
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    std::vector<TypeId> outputs_dtypes;

    const auto quant_mode =
      input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3QuantModeIndex)]
        ->GetScalarValueWithCheck<int64_t>();

    auto x_dtype =
      input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XIndex)]->GetType();
    if (input_infos[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ScalesIndex)]
          ->IsNone() &&
        quant_mode == 0) {
      (void)outputs_dtypes.emplace_back(x_dtype);
    } else {
      (void)outputs_dtypes.emplace_back(kNumberTypeInt8);
    }
    (void)outputs_dtypes.emplace_back(kNumberTypeFloat32);  // dynamic_scales
    (void)outputs_dtypes.emplace_back(kNumberTypeInt32);    // assist_info_for_combine
    (void)outputs_dtypes.emplace_back(kNumberTypeInt64);    // expert_token_nums
    (void)outputs_dtypes.emplace_back(kNumberTypeInt32);    // ep_recv_counts
    (void)outputs_dtypes.emplace_back(kNumberTypeInt32);    // tp_recv_counts
    (void)outputs_dtypes.emplace_back(kNumberTypeFloat32);  // expand_scales

    return outputs_dtypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class MoeDistributeDispatchV3CustomAscend : public AclnnCustomKernelMod {
 public:
  MoeDistributeDispatchV3CustomAscend() : AclnnCustomKernelMod("aclnnMoeDistributeDispatchV3") {}
  ~MoeDistributeDispatchV3CustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(
      stream_ptr, workspace,
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertIdsIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ScalesIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XActiveMaskIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertScalesIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ElasticInfoIndex)],
      group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_, tp_rank_id_,
      expert_shard_type_, shared_expert_num_, shared_expert_rank_num_, quant_mode_, global_bs_, expert_token_nums_type_,
      comm_alg_, zero_expert_num_, copy_expert_num_, const_expert_num_,
      outputs[static_cast<size_t>(MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3ExpandXOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3DynamicScalesOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3AssistInfoForCombineOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3ExpertTokenNumsOutputIndex)],
      outputs[static_cast<size_t>(MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3EpRecvCountsOutputIndex)],
      outputs[static_cast<size_t>(MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3TpRecvCountsOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3ExpandScalesOutputIndex)]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto group_ep = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3GroupEpIndex)]
                      ->GetOptionalValueWithCheck<std::string>();
    group_ep_ = group_ep.has_value() ? mindspore::device::ascend::OpApiUtil::GetCommName(group_ep.value())
                                     : mindspore::device::ascend::OpApiUtil::GetCommName(kHcclWorldGroup);
    ep_world_size_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3EpWorldSizeIndex)]
        ->GetValueWithCheck<int64_t>();
    ep_rank_id_ = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3EpRankIdIndex)]
                    ->GetValueWithCheck<int64_t>();
    moe_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3MoeExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    auto group_tp = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3GroupTpIndex)]
                      ->GetOptionalValueWithCheck<std::string>();
    group_tp_ = group_tp.has_value() ? mindspore::device::ascend::OpApiUtil::GetCommName(group_tp.value()) : "";

    tp_world_size_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3TpWorldSizeIndex)]
        ->GetValueWithCheck<int64_t>();
    tp_rank_id_ = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3TpRankIdIndex)]
                    ->GetValueWithCheck<int64_t>();
    expert_shard_type_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertShardTypeIndex)]
        ->GetValueWithCheck<int64_t>();
    shared_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3SharedExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    shared_expert_rank_num_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3SharedExpertRankNumIndex)]
        ->GetValueWithCheck<int64_t>();

    quant_mode_ = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3QuantModeIndex)]
                    ->GetValueWithCheck<int64_t>();
    global_bs_ = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3GlobalBsIndex)]
                   ->GetValueWithCheck<int64_t>();
    expert_token_nums_type_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertTokenNumsTypeIndex)]
        ->GetValueWithCheck<int64_t>();
    auto comm_alg = inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3CommAlgIndex)]
                      ->GetOptionalValueWithCheck<std::string>();
    comm_alg_ = (comm_alg.has_value()) ? comm_alg.value() : "";
    zero_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ZeroExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    copy_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3CopyExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    const_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ConstExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();

    GetWorkspaceForResize(
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertIdsIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ScalesIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3XActiveMaskIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ExpertScalesIndex)],
      inputs[static_cast<size_t>(MoeDistributeDispatchV3InputIndex::kMoeDistributeDispatchV3ElasticInfoIndex)],
      group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_, tp_rank_id_,
      expert_shard_type_, shared_expert_num_, shared_expert_rank_num_, quant_mode_, global_bs_, expert_token_nums_type_,
      comm_alg_, zero_expert_num_, copy_expert_num_, const_expert_num_,
      outputs[static_cast<size_t>(MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3ExpandXOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3DynamicScalesOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3AssistInfoForCombineOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3ExpertTokenNumsOutputIndex)],
      outputs[static_cast<size_t>(MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3EpRecvCountsOutputIndex)],
      outputs[static_cast<size_t>(MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3TpRecvCountsOutputIndex)],
      outputs[static_cast<size_t>(
        MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3ExpandScalesOutputIndex)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string group_ep_;
  int64_t ep_world_size_ = 0;
  int64_t ep_rank_id_ = 0;
  int64_t moe_expert_num_ = 0;
  std::string group_tp_;
  int64_t tp_world_size_ = 0;
  int64_t tp_rank_id_ = 0;
  int64_t expert_shard_type_ = 0;
  int64_t shared_expert_num_ = 0;
  int64_t shared_expert_rank_num_ = 0;
  int64_t quant_mode_ = 0;
  int64_t global_bs_ = 0;
  int64_t expert_token_nums_type_ = 0;
  std::string comm_alg_;
  int64_t zero_expert_num_ = 0;
  int64_t copy_expert_num_ = 0;
  int64_t const_expert_num_ = 0;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(moe_distribute_dispatch_v3, ms_custom_ops::MoeDistributeDispatchV3CustomOpFuncImpl,
                  ms_custom_ops::MoeDistributeDispatchV3CustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
std::vector<ms::Tensor> get_moe_distribute_v3_output_tensor(
  const ms::Tensor &x, const ms::Tensor &expert_ids, const int64_t ep_world_size, const int64_t ep_rank_id,
  const int64_t moe_expert_num, const std::optional<ms::Tensor> &scales, const std::optional<ms::Tensor> &x_active_mask,
  const std::optional<ms::Tensor> &expert_scales, const std::optional<ms::Tensor> &elastic_info,
  const std::string &group_ep, const std::optional<std::string> &group_tp, const int64_t tp_world_size,
  const int64_t tp_rank_id, const int64_t expert_shard_type, const int64_t shared_expert_num,
  const int64_t shared_expert_rank_num, const int64_t quant_mode, const int64_t global_bs,
  const int64_t expert_token_nums_type, const std::optional<std::string> &comm_alg, const int64_t zero_expert_num,
  const int64_t copy_expert_num, const int64_t const_expert_num) {
  ValidateParamsComm(shared_expert_num, shared_expert_rank_num, expert_token_nums_type, "moe_distribute_dispatch_v3");

  auto x_shape = x.shape();
  auto expert_ids_shape = expert_ids.shape();
  auto bs = x_shape[kDim0];
  auto h = x_shape[kDim1];
  auto k = expert_ids_shape[kDim1];

  const bool shared_front = (expert_shard_type == 0);
  const bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
  const bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));

  int64_t local_moe_expert_num = 1;
  int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
  int64_t a = 0;
  int64_t ep_recv_cnt_num = 0;
  if (shared_front) {
    if (ep_rank_id < shared_expert_rank_num) {
      local_moe_expert_num = 1;
      int64_t max_bs = global_bs_real / ep_world_size;
      int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;
      int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
      a = max_bs * max_shared_group_num;
    } else {
      local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
      a = global_bs_real * std::min(local_moe_expert_num, static_cast<int64_t>(k));
    }

    if (elastic_info.has_value()) {
      if ((is_shared_default) || (is_no_shared)) {
        local_moe_expert_num = std::max(
          local_moe_expert_num, static_cast<int64_t>(moe_expert_num / (ep_world_size - shared_expert_rank_num)));
        a = global_bs_real * std::min(local_moe_expert_num, static_cast<int64_t>(k));
      } else {
        int64_t max_bs = global_bs_real / ep_world_size;
        int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;
        int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
        a = std::max(
          max_bs * max_shared_group_num,
          static_cast<int64_t>(global_bs_real * std::min(moe_expert_num / (ep_world_size - shared_expert_rank_num),
                                                         static_cast<int64_t>(k))));
        local_moe_expert_num = std::max(
          local_moe_expert_num, static_cast<int64_t>(moe_expert_num / (ep_world_size - shared_expert_rank_num)));
      }
    }
  }

  if (tp_world_size == kDim2) {
    ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size;
  } else {
    ep_recv_cnt_num = ep_world_size * local_moe_expert_num;
  }

  TypeId expand_x_dtype = (!scales.has_value() && quant_mode == 0) ? x.data_type() : kNumberTypeInt8;
  TypeId dynamic_scales_dtype = kNumberTypeFloat32;
  TypeId assist_info_for_combine_dtype = kNumberTypeInt32;
  TypeId expert_token_nums_dtype = kNumberTypeInt64;
  TypeId ep_recv_counts_dtype = kNumberTypeInt32;
  TypeId tp_recv_counts_dtype = kNumberTypeInt32;
  TypeId expand_scales_dtype = kNumberTypeFloat32;

  ShapeVector expand_x_shape = (tp_world_size == 0) ? ShapeVector{a, h} : ShapeVector{a * tp_world_size, h};
  ShapeVector dynamic_scales_shape =
    (tp_world_size == 0) ? ShapeVector{static_cast<int64_t>(1)} : ShapeVector{a * tp_world_size};
  ShapeVector assist_info_for_combine_shape = {std::max(bs * k, a * 128)};
  ShapeVector expert_token_nums_shape = {local_moe_expert_num};
  if (expert_scales.has_value()) {
    ep_recv_cnt_num = ep_world_size * local_moe_expert_num +
                      2 * global_bs_real * k * (ep_world_size / 8);  // 2: 2 buffer, 8 ranknum per server
  }
  ShapeVector ep_recv_counts_shape = {ep_recv_cnt_num};
  ShapeVector tp_recv_counts_shape = {tp_world_size};
  ShapeVector expand_scales_shape = {a};

  std::vector<ms::Tensor> outputs = {
    ms::Tensor(expand_x_dtype, expand_x_shape),
    ms::Tensor(dynamic_scales_dtype, dynamic_scales_shape),
    ms::Tensor(assist_info_for_combine_dtype, assist_info_for_combine_shape),
    ms::Tensor(expert_token_nums_dtype, expert_token_nums_shape),
    ms::Tensor(ep_recv_counts_dtype, ep_recv_counts_shape),
    ms::Tensor(tp_recv_counts_dtype, tp_recv_counts_shape),
    ms::Tensor(expand_scales_dtype, expand_scales_shape),
  };

  return outputs;
}

std::vector<ms::Tensor> moe_distribute_dispatch_v3_custom(
  const ms::Tensor &x, const ms::Tensor &expert_ids, const int64_t ep_world_size, const int64_t ep_rank_id,
  const int64_t moe_expert_num, const std::optional<ms::Tensor> &scales, const std::optional<ms::Tensor> &x_active_mask,
  const std::optional<ms::Tensor> &expert_scales, const std::optional<ms::Tensor> &elastic_info,
  const std::string &group_ep, const std::optional<std::string> &group_tp, const int64_t tp_world_size,
  const int64_t tp_rank_id, const int64_t expert_shard_type, const int64_t shared_expert_num,
  const int64_t shared_expert_rank_num, const int64_t quant_mode, const int64_t global_bs,
  const int64_t expert_token_nums_type, const std::optional<std::string> &comm_alg, const int64_t zero_expert_num,
  const int64_t copy_expert_num, const int64_t const_expert_num) {
  std::vector<ms::Tensor> outputs = get_moe_distribute_v3_output_tensor(
    x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, elastic_info,
    group_ep, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num,
    quant_mode, global_bs, expert_token_nums_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnMoeDistributeDispatchV3");

  auto new_group_ep = (group_ep == "") ? mindspore::device::ascend::OpApiUtil::GetCommName(kHcclWorldGroup)
                                       : mindspore::device::ascend::OpApiUtil::GetCommName(group_ep);
  auto group_tp_str = (group_tp.has_value() ? group_tp.value() : "");
  auto new_group_tp = (group_tp_str.empty()) ? "" : mindspore::device::ascend::OpApiUtil::GetCommName(group_tp_str);
  auto new_comm_alg = (comm_alg.has_value() ? comm_alg.value() : "");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
    aclnnMoeDistributeDispatchV3, x, expert_ids, scales, x_active_mask, expert_scales, elastic_info, new_group_ep,
    ep_world_size, ep_rank_id, moe_expert_num, new_group_tp, tp_world_size, tp_rank_id, expert_shard_type,
    shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, new_comm_alg,
    zero_expert_num, copy_expert_num, const_expert_num, outputs[kDim0], outputs[kDim1], outputs[kDim2], outputs[kDim3],
    outputs[kDim4], outputs[kDim5], outputs[kDim6]));
  // only set tensor.
  runner->Run({x, expert_ids, GetTensorOrEmpty(scales), GetTensorOrEmpty(x_active_mask),
               GetTensorOrEmpty(expert_scales), GetTensorOrEmpty(elastic_info)},
              outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_moe_distribute_dispatch_v3(
  const ms::Tensor &x, const ms::Tensor &expert_ids, const int64_t ep_world_size, const int64_t ep_rank_id,
  const int64_t moe_expert_num, const std::optional<ms::Tensor> &scales, const std::optional<ms::Tensor> &x_active_mask,
  const std::optional<ms::Tensor> &expert_scales, const std::optional<ms::Tensor> &elastic_info,
  const std::string &group_ep, const std::optional<std::string> &group_tp, const int64_t tp_world_size,
  const int64_t tp_rank_id, const int64_t expert_shard_type, const int64_t shared_expert_num,
  const int64_t shared_expert_rank_num, const int64_t quant_mode, const int64_t global_bs,
  const int64_t expert_token_nums_type, const std::optional<std::string> &comm_alg, const int64_t zero_expert_num,
  const int64_t copy_expert_num, const int64_t const_expert_num) {
  return ms::pynative::PyboostRunner::Call<static_cast<size_t>(
    ms_custom_ops::MoeDistributeDispatchV3OutputIndex::kMoeDistributeDispatchV3OutputNums)>(
    ms_custom_ops::moe_distribute_dispatch_v3_custom, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, scales,
    x_active_mask, expert_scales, elastic_info, group_ep, group_tp, tp_world_size, tp_rank_id, expert_shard_type,
    shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_alg, zero_expert_num,
    copy_expert_num, const_expert_num);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("moe_distribute_dispatch_v3", &pyboost_moe_distribute_dispatch_v3, "MoeDistributeDispatchV3",
        pybind11::arg("x"), pybind11::arg("expert_ids"), pybind11::arg("ep_world_size"), pybind11::arg("ep_rank_id"),
        pybind11::arg("moe_expert_num"), pybind11::arg("scales") = std::nullopt,
        pybind11::arg("x_active_mask") = std::nullopt, pybind11::arg("expert_scales") = std::nullopt,
        pybind11::arg("elastic_info") = std::nullopt, pybind11::arg("group_ep") = "",
        pybind11::arg("group_tp") = std::nullopt, pybind11::arg("tp_world_size") = 0, pybind11::arg("tp_rank_id") = 0,
        pybind11::arg("expert_shard_type") = 0, pybind11::arg("shared_expert_num") = 1,
        pybind11::arg("shared_expert_rank_num") = 0, pybind11::arg("quant_mode") = 0, pybind11::arg("global_bs") = 0,
        pybind11::arg("expert_token_nums_type") = 1, pybind11::arg("comm_alg") = std::nullopt,
        pybind11::arg("zero_expert_num") = 0, pybind11::arg("copy_expert_num") = 0,
        pybind11::arg("const_expert_num") = 0);
}
