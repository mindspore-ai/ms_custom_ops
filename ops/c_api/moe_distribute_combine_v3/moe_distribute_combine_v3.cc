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

#include <map>
#include <string>
#include <vector>
#include <set>
#include <memory>
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"
#include "ops/c_api/utils/check_utils.h"

namespace ms_custom_ops {
enum class MoeDistributeCombineInputIndex : size_t {
  kMoeDistributeCombineExpandXIndex = 0,
  kMoeDistributeCombineExpertIdsIndex,
  kMoeDistributeCombineAssistInfoForCombineIndex,
  kMoeDistributeCombineEpSendCountsIndex,
  kMoeDistributeCombineExpertScalesIndex,
  kMoeDistributeCombineEpWorldSizeIndex,
  kMoeDistributeCombineEpRankIdIndex,
  kMoeDistributeCombineMoeExpertNumIndex,
  kMoeDistributeCombineGroupEpIndex,
  kMoeDistributeCombineTpSendCountsIndex,
  kMoeDistributeCombineXActiveMaskIndex,
  kMoeDistributeCombineActivationScaleIndex,
  kMoeDistributeCombineWeightScaleIndex,
  kMoeDistributeCombineGroupListIndex,
  kMoeDistributeCombineExpandScalesIndex,
  kMoeDistributeCombineSharedExpertXIndex,
  kMoeDistributeCombineElasticInfoIndex,
  kMoeDistributeCombineOriXIndex,
  kMoeDistributeCombineConstExpertAlpha1Index,
  kMoeDistributeCombineConstExpertAlpha2Index,
  kMoeDistributeCombineConstExpertVIndex,
  kMoeDistributeCombineGroupTpIndex,
  kMoeDistributeCombineTpWorldSizeIndex,
  kMoeDistributeCombineTpRankIdIndex,
  kMoeDistributeCombineExpertShardTypeIndex,
  kMoeDistributeCombineSharedExpertNumIndex,
  kMoeDistributeCombineSharedExpertRankNumIndex,
  kMoeDistributeCombineGlobalBsIndex,
  kMoeDistributeCombineOutDtypeIndex,
  kMoeDistributeCombineCommQuantModeIndex,
  kMoeDistributeCombineGroupListTypeIndex,
  kMoeDistributeCombineCommAlgIndex,
  kMoeDistributeCombineZeroExpertNumIndex,
  kMoeDistributeCombineCopyExpertNumIndex,
  kMoeDistributeCombineConstExpertNumIndex,
  kMoeDistributeCombineInputsNum,
};

enum class MoeDistributeCombineOutputIndex : size_t {
  kMoeDistributeCombineXOutIndex = 0,
  kMoeDistributeCombineOutputsNum,
};

class OPS_API MoeDistributeCombineCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    // (max(tpWorldSize, 1) * A , H)
    auto expand_x_shape =
      input_infos[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpandXIndex)]->GetShape();
    // (Bs, K)
    auto expert_ids_shape =
      input_infos[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpertIdsIndex)]->GetShape();
    if (IsDynamicRank(expand_x_shape) || IsDynamicRank(expert_ids_shape)) {
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }
    if (expand_x_shape.size() != kDim2 || expert_ids_shape.size() != kDim2) {
      MS_LOG(EXCEPTION)
        << "For 'MoeDistributeCombine', the dimension of 'expand_x' and 'expert_ids' should be 2, but got "
        << expand_x_shape.size() << " and " << expert_ids_shape.size();
    }
    // (Bs, H)
    ShapeVector out_shape;
    out_shape.push_back(expert_ids_shape[kDim0]);
    out_shape.push_back(expand_x_shape[kDim1]);
    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
    auto expand_x_dtype =
      input_infos[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpandXIndex)]->GetType();
    CheckAndConvertUtils::CheckTypeIdValid("expand_x", expand_x_dtype, valid_types, op_name);
    return {expand_x_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class MoeDistributeCombineCustomAscend : public AclnnCustomKernelMod {
 public:
  MoeDistributeCombineCustomAscend() : AclnnCustomKernelMod("aclnnMoeDistributeCombineV3") {}
  ~MoeDistributeCombineCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace,
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpandXIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpertIdsIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineAssistInfoForCombineIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineEpSendCountsIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpertScalesIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineTpSendCountsIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineXActiveMaskIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineActivationScaleIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineWeightScaleIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineGroupListIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpandScalesIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineSharedExpertXIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineElasticInfoIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineOriXIndex)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertAlpha1Index)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertAlpha2Index)],
          inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertVIndex)],
          group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_, tp_rank_id_,
          expert_shard_type_, shared_expert_num_, shared_expert_rank_num_, global_bs_, out_dtype_, common_quant_mode_,
          group_list_type_, comm_alg_, zero_expert_num_, copy_expert_num_, const_expert_num_,
          outputs[static_cast<size_t>(MoeDistributeCombineOutputIndex::kMoeDistributeCombineXOutIndex)]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    ep_world_size_ = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineEpWorldSizeIndex)]
                       ->GetValueWithCheck<int64_t>();
    ep_rank_id_ = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineEpRankIdIndex)]
                    ->GetValueWithCheck<int64_t>();
    moe_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineMoeExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    auto group_ep = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineGroupEpIndex)]
                      ->GetValueWithCheck<std::string>();
    group_ep_ = group_ep.empty() ? mindspore::device::ascend::OpApiUtil::GetCommName(kHcclWorldGroup)
                                 : mindspore::device::ascend::OpApiUtil::GetCommName(group_ep);
    auto group_tp = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineGroupTpIndex)]
                      ->GetOptionalValueWithCheck<std::string>();
    group_tp_ = group_tp.has_value() ? mindspore::device::ascend::OpApiUtil::GetCommName(group_tp.value()) : "";
    tp_world_size_ = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineTpWorldSizeIndex)]
                       ->GetValueWithCheck<int64_t>();
    tp_rank_id_ = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineTpRankIdIndex)]
                    ->GetValueWithCheck<int64_t>();
    expert_shard_type_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpertShardTypeIndex)]
        ->GetValueWithCheck<int64_t>();
    shared_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineSharedExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    shared_expert_rank_num_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineSharedExpertRankNumIndex)]
        ->GetValueWithCheck<int64_t>();
    global_bs_ = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineGlobalBsIndex)]
                   ->GetValueWithCheck<int64_t>();
    out_dtype_ = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineOutDtypeIndex)]
                   ->GetValueWithCheck<int64_t>();
    common_quant_mode_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineCommQuantModeIndex)]
        ->GetValueWithCheck<int64_t>();
    group_list_type_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineGroupListTypeIndex)]
        ->GetValueWithCheck<int64_t>();
    auto comm_alg = inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineCommAlgIndex)]
                      ->GetOptionalValueWithCheck<std::string>();
    comm_alg_ = comm_alg.has_value() ? comm_alg.value() : "";
    zero_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineZeroExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    copy_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineCopyExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();
    const_expert_num_ =
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertNumIndex)]
        ->GetValueWithCheck<int64_t>();

    GetWorkspaceForResize(
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpandXIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpertIdsIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineAssistInfoForCombineIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineEpSendCountsIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpertScalesIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineTpSendCountsIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineXActiveMaskIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineActivationScaleIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineWeightScaleIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineGroupListIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineExpandScalesIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineSharedExpertXIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineElasticInfoIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineOriXIndex)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertAlpha1Index)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertAlpha2Index)],
      inputs[static_cast<size_t>(MoeDistributeCombineInputIndex::kMoeDistributeCombineConstExpertVIndex)], group_ep_,
      ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_, tp_rank_id_, expert_shard_type_,
      shared_expert_num_, shared_expert_rank_num_, global_bs_, out_dtype_, common_quant_mode_, group_list_type_,
      comm_alg_, zero_expert_num_, copy_expert_num_, const_expert_num_,
      outputs[static_cast<size_t>(MoeDistributeCombineOutputIndex::kMoeDistributeCombineXOutIndex)]);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  int64_t ep_world_size_ = 0;
  int64_t ep_rank_id_ = 0;
  int64_t moe_expert_num_ = 0;
  std::string group_ep_;
  std::string group_tp_;
  int64_t tp_world_size_ = 0;
  int64_t tp_rank_id_ = 0;
  int64_t expert_shard_type_ = 0;
  int64_t shared_expert_num_ = 1;
  int64_t shared_expert_rank_num_ = 0;
  int64_t global_bs_ = 0;
  int64_t out_dtype_ = 0;
  int64_t common_quant_mode_ = 0;
  int64_t group_list_type_ = 0;
  std::string comm_alg_;
  int64_t zero_expert_num_ = 0;
  int64_t copy_expert_num_ = 0;
  int64_t const_expert_num_ = 0;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(moe_distribute_combine_v3, ms_custom_ops::MoeDistributeCombineCustomOpFuncImpl,
                  ms_custom_ops::MoeDistributeCombineCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
ms::Tensor moe_distribute_combine_v3_custom(
  const ms::Tensor &expand_x, const ms::Tensor &expert_ids, const ms::Tensor &assist_info_for_combine,
  const ms::Tensor &ep_send_count, const ms::Tensor &expert_scale, const int64_t ep_world_size,
  const int64_t ep_rank_id, const int64_t moe_expert_num, const std::string &group_ep,
  const std::optional<ms::Tensor> &tp_send_count, const std::optional<ms::Tensor> &x_activate_mask,
  const std::optional<ms::Tensor> &activation_scale, const std::optional<ms::Tensor> &weight_scale,
  const std::optional<ms::Tensor> &group_list, const std::optional<ms::Tensor> &expand_scales,
  const std::optional<ms::Tensor> &shared_expert_x, const std::optional<ms::Tensor> &elastic_info,
  const std::optional<ms::Tensor> &ori_x, const std::optional<ms::Tensor> &const_expert_alpha1,
  const std::optional<ms::Tensor> &const_expert_alpha2, const std::optional<ms::Tensor> &const_expert_v,
  const std::optional<std::string> &group_tp, const int64_t tp_world_size, const int64_t tp_rank_id,
  const int64_t expert_shard_type, const int64_t shared_expert_num, const int64_t shared_expert_rank_num,
  const int64_t global_bs, const int64_t out_dtype, const int64_t common_quant_mode, const int64_t group_list_type,
  const std::optional<std::string> &comm_alg, const int64_t zero_expert_num, const int64_t copy_expert_num,
  const int64_t const_expert_num) {
  // (max(tpWorldSize, 1) * A , H)
  auto expand_x_shape = expand_x.shape();
  // (Bs, K)
  auto expert_ids_shape = expert_ids.shape();

  if (expand_x_shape.size() != kDim2 || expert_ids_shape.size() != kDim2) {
    MS_LOG(EXCEPTION)
      << "For 'MoeDistributeCombine', the dimension of 'expand_x' and 'expert_ids' should be 2, but got "
      << expand_x_shape.size() << " and " << expert_ids_shape.size();
  }
  // (Bs, H)
  ShapeVector out_shape;
  out_shape.push_back(expert_ids_shape[kDim0]);
  out_shape.push_back(expand_x_shape[kDim1]);
  auto out = ms::Tensor(expand_x.data_type(), out_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnMoeDistributeCombineV3");
  auto new_group_ep = group_ep.empty() ? mindspore::device::ascend::OpApiUtil::GetCommName(kHcclWorldGroup)
                                       : mindspore::device::ascend::OpApiUtil::GetCommName(group_ep);
  auto group_tp_str = group_tp.has_value() ? group_tp.value() : "";
  auto new_group_tp = group_tp_str.empty() ? "" : mindspore::device::ascend::OpApiUtil::GetCommName(group_tp_str);
  auto new_comm_alg = comm_alg.has_value() ? comm_alg.value() : "";
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
    aclnnMoeDistributeCombineV3, expand_x, expert_ids, assist_info_for_combine, ep_send_count, expert_scale,
    tp_send_count, x_activate_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x,
    elastic_info, ori_x, const_expert_alpha1, const_expert_alpha2, const_expert_v, new_group_ep, ep_world_size,
    ep_rank_id, moe_expert_num, new_group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num,
    shared_expert_rank_num, global_bs, out_dtype, common_quant_mode, group_list_type, new_comm_alg, zero_expert_num,
    copy_expert_num, const_expert_num, out));
  runner->Run({expand_x, expert_ids, assist_info_for_combine, ep_send_count, expert_scale,
               GetTensorOrEmpty(tp_send_count), GetTensorOrEmpty(x_activate_mask), GetTensorOrEmpty(weight_scale),
               GetTensorOrEmpty(group_list), GetTensorOrEmpty(expand_scales), GetTensorOrEmpty(shared_expert_x),
               GetTensorOrEmpty(elastic_info), GetTensorOrEmpty(ori_x), GetTensorOrEmpty(const_expert_alpha1),
               GetTensorOrEmpty(const_expert_alpha2), GetTensorOrEmpty(const_expert_v)},
              {out});
  return out;
}
}  // namespace ms_custom_ops

auto pyboost_moe_distribute_combine_v3(
  const ms::Tensor &expand_x, const ms::Tensor &expert_ids, const ms::Tensor &assist_info_for_combine,
  const ms::Tensor &ep_send_count, const ms::Tensor &expert_scale, const int64_t ep_world_size,
  const int64_t ep_rank_id, const int64_t moe_expert_num, const std::string &group_ep,
  const std::optional<ms::Tensor> &tp_send_count, const std::optional<ms::Tensor> &x_activate_mask,
  const std::optional<ms::Tensor> &activation_scale, const std::optional<ms::Tensor> &weight_scale,
  const std::optional<ms::Tensor> &group_list, const std::optional<ms::Tensor> &expand_scales,
  const std::optional<ms::Tensor> &shared_expert_x, const std::optional<ms::Tensor> &elastic_info,
  const std::optional<ms::Tensor> &ori_x, const std::optional<ms::Tensor> &const_expert_alpha1,
  const std::optional<ms::Tensor> &const_expert_alpha2, const std::optional<ms::Tensor> &const_expert_v,
  const std::optional<std::string> &group_tp, const int64_t tp_world_size, const int64_t tp_rank_id,
  const int64_t expert_shard_type, const int64_t shared_expert_num, const int64_t shared_expert_rank_num,
  const int64_t global_bs, const int64_t out_dtype, const int64_t common_quant_mode, const int64_t group_list_type,
  const std::optional<std::string> &comm_alg, const int64_t zero_expert_num, const int64_t copy_expert_num,
  const int64_t const_expert_num) {
  return ms::pynative::PyboostRunner::Call<static_cast<size_t>(
    ms_custom_ops::MoeDistributeCombineOutputIndex::kMoeDistributeCombineOutputsNum)>(
    ms_custom_ops::moe_distribute_combine_v3_custom, expand_x, expert_ids, assist_info_for_combine, ep_send_count,
    expert_scale, ep_world_size, ep_rank_id, moe_expert_num, group_ep, tp_send_count, x_activate_mask, activation_scale,
    weight_scale, group_list, expand_scales, shared_expert_x, elastic_info, ori_x, const_expert_alpha1,
    const_expert_alpha2, const_expert_v, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num,
    shared_expert_rank_num, global_bs, out_dtype, common_quant_mode, group_list_type, comm_alg, zero_expert_num,
    copy_expert_num, const_expert_num);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("moe_distribute_combine_v3", &pyboost_moe_distribute_combine_v3, "MoeDistributeCombineV3",
        pybind11::arg("expand_x"), pybind11::arg("expert_ids"), pybind11::arg("assist_info_for_combine"),
        pybind11::arg("ep_send_counts"), pybind11::arg("expert_scales"), pybind11::arg("ep_world_size"),
        pybind11::arg("ep_rank_id"), pybind11::arg("moe_expert_num"), pybind11::arg("group_ep"),
        pybind11::arg("tp_send_counts") = std::nullopt, pybind11::arg("x_active_mask") = std::nullopt,
        pybind11::arg("activation_scale") = std::nullopt, pybind11::arg("weight_scale") = std::nullopt,
        pybind11::arg("group_list") = std::nullopt, pybind11::arg("expand_scales") = std::nullopt,
        pybind11::arg("shared_expert_x") = std::nullopt, pybind11::arg("elastic_info") = std::nullopt,
        pybind11::arg("ori_x") = std::nullopt, pybind11::arg("const_expert_alpha1") = std::nullopt,
        pybind11::arg("const_expert_alpha2") = std::nullopt, pybind11::arg("const_expert_v") = std::nullopt,
        pybind11::arg("group_tp") = std::nullopt, pybind11::arg("tp_world_size") = 0, pybind11::arg("tp_rank_id") = 0,
        pybind11::arg("expert_shard_type") = 0, pybind11::arg("shared_expert_num") = 1,
        pybind11::arg("shared_expert_rank_num") = 0, pybind11::arg("global_bs") = 0, pybind11::arg("out_dtype") = 0,
        pybind11::arg("common_quant_mode") = 0, pybind11::arg("group_list_type") = 0,
        pybind11::arg("comm_alg") = std::nullopt, pybind11::arg("zero_expert_num") = 0,
        pybind11::arg("copy_expert_num") = 0, pybind11::arg("const_expert_num") = 0);
}
