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

#include <map>
#include <string>
#include <vector>
#include "ops/c_api/grouped_matmul_v4_cops/grouped_matmul_v4_cops_func_impl.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"

constexpr auto kGroupedMatmulV4CopsName = "grouped_matmul_v4_cops";
constexpr size_t kListTensorInputNums = 12;

namespace ms_custom_ops {
std::vector<std::vector<KernelTensor *>> DealWithListTensors(const std::vector<int64_t> &group_info,
                                                             const std::vector<int64_t> &start_idxs,
                                                             const std::vector<KernelTensor *> &inputs) {
  std::vector<std::vector<KernelTensor *>> list_inputs{};
  for (size_t i = 0; i < kListTensorInputNums; i++) {
    std::vector<KernelTensor *> input_i{};
    if (group_info[i] > 0) {
      input_i.assign(inputs.begin() + start_idxs[i], inputs.begin() + start_idxs[i + 1]);
    }
    (void)list_inputs.emplace_back(std::move(input_i));
  }
  return list_inputs;
}

std::vector<int64_t> ComputeStartIdxsFromGroupInfo(const std::vector<int64_t> &group_info) {
  std::vector<int64_t> start_idxs{0};
  int64_t cur_end_idx = 0;
  for (size_t i = 0; i < group_info.size(); ++i) {
    cur_end_idx += (group_info[i] == 0 ? 1 : group_info[i]);
    start_idxs.push_back(cur_end_idx);
  }
  return start_idxs;
}

static inline void UnifyWeightShape(const std::vector<KernelTensor *> &ori_weights,
                                    std::vector<std::shared_ptr<KernelTensor>> *new_weights_shared_ptr,
                                    std::vector<KernelTensor *> *new_weights_raw_ptr) {
  for (const auto &w : ori_weights) {
    if (w->dtype_id() == kNumberTypeInt4) {
      const auto &storage_info = w->tensor_storage_info();
      if (storage_info != nullptr && !storage_info->is_contiguous) {
        MS_LOG(EXCEPTION) << "Currently, " << kGroupedMatmulV4CopsName
                          << " does not support noncontiguous input tensor for int4 quant, "
                          << "but got noncontiguous input tensor: " << w->ToString()
                          << ", storage info: " << storage_info->ToString();
      }
      auto new_w = w->CloneKernelTensor();
      auto w_shape = w->GetShapeVector();
      w_shape.back() *= kNumber2;
      new_w->SetShapeVector(w_shape);
      new_weights_shared_ptr->emplace_back(new_w);
      new_weights_raw_ptr->emplace_back(new_w.get());
    } else {
      new_weights_raw_ptr->emplace_back(w);
    }
  }
}

class GroupedMatmulV4CopsAscend : public AclnnCustomKernelMod {
 public:
  GroupedMatmulV4CopsAscend() : AclnnCustomKernelMod("aclnnGroupedMatmulV4") {}
  ~GroupedMatmulV4CopsAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    auto list_inputs = DealWithListTensors(group_info_, start_idxs_, inputs);
    const auto &group_list = inputs[start_idxs_[group_list_idx_]];

    std::vector<std::shared_ptr<KernelTensor>> new_weights;
    std::vector<KernelTensor *> new_weights_raw;
    UnifyWeightShape(list_inputs[kIndex1], &new_weights, &new_weights_raw);

    RunOp(stream_ptr, workspace, list_inputs[kIndex0], new_weights_raw, list_inputs[kIndex2], list_inputs[kIndex3],
          list_inputs[kIndex4], list_inputs[kIndex5], list_inputs[kIndex6], list_inputs[kIndex7], group_list,
          list_inputs[kIndex9], list_inputs[kIndex10], list_inputs[kIndex11], split_item_, group_type_,
          group_list_type_, act_type_, outputs, activation_feature_out_, dyn_quant_scale_out_);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    group_info_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(kGroupInfo));
    start_idxs_ = ComputeStartIdxsFromGroupInfo(group_info_);

    auto list_inputs = DealWithListTensors(group_info_, start_idxs_, inputs);
    const auto &group_list = inputs[start_idxs_[group_list_idx_]];
    const auto split_item_idx = start_idxs_.back();
    split_item_ = inputs.at(split_item_idx)->GetValueWithCheck<int64_t>();
    group_type_ = inputs.at(split_item_idx + kIndex1)->GetValueWithCheck<int64_t>();
    group_list_type_ = inputs.at(split_item_idx + kIndex2)->GetValueWithCheck<int64_t>();
    act_type_ = inputs.at(split_item_idx + kIndex3)->GetValueWithCheck<int64_t>();
    weight_format_ = inputs.at(split_item_idx + kIndex4)->GetValueWithCheck<std::string>();

    std::vector<std::shared_ptr<KernelTensor>> new_weights;
    std::vector<KernelTensor *> new_weights_raw;
    UnifyWeightShape(list_inputs[kIndex1], &new_weights, &new_weights_raw);

    if (weight_format_ == kFractalNzFormat) {
      for (auto &w_tensor : new_weights_raw) {
        w_tensor->set_format(mindspore::Format::FRACTAL_NZ);
        if (w_tensor->tensor_storage_info() != nullptr) {
          MS_LOG(EXCEPTION) << "For " << kGroupedMatmulV4CopsName
                            << ", FRACTAL_NZ is not support when storage_info is not nullptr";
        }
        auto nd_shape = w_tensor->GetShapeVector();
        auto storage_info = GetNZFormatStorageInfo(nd_shape, w_tensor->dtype_id());
        w_tensor->set_tensor_storage_info(storage_info);
      }
    }

    GetWorkspaceForResize(list_inputs[kIndex0], new_weights_raw, list_inputs[kIndex2], list_inputs[kIndex3],
                          list_inputs[kIndex4], list_inputs[kIndex5], list_inputs[kIndex6], list_inputs[kIndex7],
                          group_list, list_inputs[kIndex9], list_inputs[kIndex10], list_inputs[kIndex11], split_item_,
                          group_type_, group_list_type_, act_type_, outputs, activation_feature_out_,
                          dyn_quant_scale_out_);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  const size_t group_list_idx_{8};
  int64_t act_type_{0};
  int64_t group_list_type_{0};
  int64_t group_type_{0};
  int64_t split_item_{0};
  std::string weight_format_{"ND"};
  std::vector<int64_t> group_info_{};
  std::vector<int64_t> start_idxs_{};
  const std::vector<KernelTensor *> activation_feature_out_{};
  const std::vector<KernelTensor *> dyn_quant_scale_out_{};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(grouped_matmul_v4_cops, ms_custom_ops::GroupedMatmulV4CopsFuncImpl,
                  ms_custom_ops::GroupedMatmulV4CopsAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
TypeIdList InferOutputTypes(const std::vector<ms::Tensor> &x, const std::vector<ms::Tensor> &weight,
                            const std::optional<std::vector<ms::Tensor>> &scale,
                            const std::optional<int64_t> &output_dtype) {
  TypeId x_type = x[0].data_type();
  TypeId w_type = weight[0].data_type();
  TypeIdList output_types;
  if (x_type == kNumberTypeInt8 && w_type == kNumberTypeInt4) {
    TypeId output_type = kNumberTypeBFloat16;
    if (output_dtype.has_value()) {
      output_type = static_cast<TypeId>(output_dtype.value());
      static std::set<TypeId> valid_dtype_set = {kNumberTypeFloat16, kNumberTypeBFloat16};
      if (valid_dtype_set.find(output_type) == valid_dtype_set.end()) {
        MS_EXCEPTION(ValueError) << "For " << kGroupedMatmulV4CopsName
                                 << " with A8W4, the output type must be in [Float16, BFloat16], but got "
                                 << TypeIdToString(output_type);
      }
    }
    std::transform(x.begin(), x.end(), std::back_inserter(output_types),
                   [output_type](const ms::Tensor &info) { return output_type; });
  } else if (!scale.has_value()) {
    auto out_type = x_type == kNumberTypeInt8 ? kNumberTypeInt32 : x_type;
    std::transform(x.begin(), x.end(), std::back_inserter(output_types),
                   [=](const ms::Tensor &info) { return out_type; });
  } else {
    const auto &scale_tensors = scale.value();
    TypeId scale_type = scale_tensors[0].data_type();
    if (scale_type == kNumberTypeUInt64) {
      std::transform(x.begin(), x.end(), std::back_inserter(output_types),
                     [](const ms::Tensor &info) { return kNumberTypeInt8; });
    } else if (scale_type == kNumberTypeBFloat16) {
      std::transform(x.begin(), x.end(), std::back_inserter(output_types),
                     [](const ms::Tensor &info) { return kNumberTypeBFloat16; });
    } else if (scale_type == kNumberTypeFloat32) {
      std::transform(x.begin(), x.end(), std::back_inserter(output_types),
                     [](const ms::Tensor &info) { return kNumberTypeFloat16; });
    } else {
      MS_EXCEPTION(ValueError) << "For " << kGroupedMatmulV4CopsName
                               << ", the scale only support Uint16, BFloat16 and Float32, but got " << scale_type;
    }
  }
  return output_types;
}

ShapeArray InferOutputShapes(const std::vector<ms::Tensor> &x, const std::vector<ms::Tensor> &weight,
                             const std::optional<ms::Tensor> &group_list, const int64_t &group_type) {
  ShapeArray output_shapes;
  if (group_type == -1) {
    if (MS_UNLIKELY(x.size() != weight.size())) {
      MS_EXCEPTION(ValueError) << "For " << kGroupedMatmulV4CopsName
                               << ", when group_type is -1 and split_item is 0, x's size "
                               << "should be equal to weight, but got " << x.size() << " and " << weight.size();
    }
    for (size_t i = 0; i < x.size(); i++) {
      const auto &x_shape = x[i].shape();
      const auto &w_shape = weight[i].shape();
      auto res_shape = x_shape;
      res_shape.back() = w_shape.back();
      (void)output_shapes.emplace_back(std::move(res_shape));
    }
  } else {
    if (MS_UNLIKELY(x.size() != kDim1 || weight.size() != kDim1)) {
      MS_EXCEPTION(ValueError) << "For " << kGroupedMatmulV4CopsName
                               << ", when split_item is 3. the size of x and weight should "
                               << "both be 1, but got x's size " << x.size() << ", and weight's size " << weight.size();
    }
    const auto &x_shape = x[0].shape();
    const auto &w_shape = weight[0].shape();
    auto m = x_shape[x_shape.size() - kIndex2];
    auto n = w_shape.back();
    bool is_int4 = weight[0].data_type() == kNumberTypeInt4;
    if (is_int4) {
      n = n << 1;
    }
    int64_t group_list_size = group_list.has_value() ? group_list.value().shape()[0] : 1;
    std::vector<int64_t> res_shape;
    if (group_type == 0) {
      res_shape = std::vector<int64_t>{m, n};
    } else if (group_type == 1) {
      res_shape = std::vector<int64_t>{group_list_size, m, n};
    }
    (void)output_shapes.emplace_back(std::move(res_shape));
  }
  return output_shapes;
}

std::vector<ms::Tensor> CreateEmptyOutputTensors(TypeIdList &output_types, const ShapeArray &output_shapes) {
  if (output_types.size() != output_shapes.size()) {
    MS_EXCEPTION(ValueError) << "For " << kGroupedMatmulV4CopsName
                             << ", the number of output types must be equal to the number of output shapes, "
                             << "but got output types size: " << output_types.size()
                             << ", output shapes size: " << output_shapes.size();
  }
  std::vector<ms::Tensor> empty_tensors;
  for (size_t i = 0; i < output_types.size(); i++) {
    auto out_shape = output_shapes[i];
    TypeId out_type = static_cast<TypeId>(output_types[i]);
    auto out_tensor = ms::Tensor(out_type, out_shape);
    (void)empty_tensors.emplace_back(out_tensor);
  }
  return empty_tensors;
}

std::vector<ms::Tensor> GetOptionalTensorList(const std::optional<std::vector<ms::Tensor>> &tensor_list) {
  return tensor_list.has_value() ? tensor_list.value() : std::vector<ms::Tensor>();
}

void GetFlattenInputs(const std::vector<ms::Tensor> &tensor_list, std::vector<ms::Tensor> &inputs) {
  for (auto &tensor : tensor_list) {
    (void)inputs.emplace_back(tensor);
  }
}

void UnifyWeightShape(const std::vector<ms::Tensor> &ori_weights, std::vector<ms::Tensor> *new_weights) {
  for (const auto &ori_weight : ori_weights) {
    if (ori_weight.data_type() == kNumberTypeInt4) {
      MS_EXCEPTION_IF_NULL(ori_weight.tensor());
      const auto &storage_info = ori_weight.tensor()->storage_info();
      if (storage_info != nullptr && !storage_info->is_contiguous) {
        MS_LOG(EXCEPTION) << "Currently, " << kGroupedMatmulV4CopsName
                          << " does not support noncontiguous input tensor for int4 quant, "
                          << "but got noncontiguous input tensor: " << ori_weight.tensor()->ToString()
                          << ", storage info: " << storage_info->ToString();
      }
      auto new_weight = ms::Tensor(ori_weight.data_type(), ori_weight.shape());
      new_weight.AssignTensor(ori_weight);
      auto ori_weight_shape = ori_weight.shape();
      ori_weight_shape.back() *= kNumber2;
      new_weight.tensor()->set_shape(ori_weight_shape);
      (void)new_weights->emplace_back(std::move(new_weight));
    } else {
      (void)new_weights->emplace_back(ori_weight);
    }
  }
}

std::vector<ms::Tensor> grouped_matmul_v4_cops_custom(
  const std::vector<ms::Tensor> &x, const std::vector<ms::Tensor> &weight,
  const std::optional<std::vector<ms::Tensor>> &bias, const std::optional<std::vector<ms::Tensor>> &scale,
  const std::optional<std::vector<ms::Tensor>> &offset, const std::optional<std::vector<ms::Tensor>> &antiquant_scale,
  const std::optional<std::vector<ms::Tensor>> &antiquant_offset,
  const std::optional<std::vector<ms::Tensor>> &per_token_scale, const std::optional<ms::Tensor> &group_list,
  const std::optional<std::vector<ms::Tensor>> &activation_input,
  const std::optional<std::vector<ms::Tensor>> &activation_quant_scale,
  const std::optional<std::vector<ms::Tensor>> &activation_quant_offset, const int64_t &split_item,
  const int64_t &group_type, const int64_t &group_list_type, const int64_t &act_type, const std::string &weight_format,
  const std::optional<int64_t> &output_dtype) {
  auto output_types = InferOutputTypes(x, weight, scale, output_dtype);
  auto output_shapes = InferOutputShapes(x, weight, group_list, group_type);
  auto outputs = CreateEmptyOutputTensors(output_types, output_shapes);

  // Because copying the tensor is required under UnifyWeightShape, the address needs to be allocated in advance.
  auto device_type = mindspore::DeviceManagerConf::GetInstance()->device_type();
  auto device_context = mindspore::runtime::OpRunner::GetDeviceContext(device_type);
  auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  for (size_t i = 0; i < weight.size(); i++) {
    mindspore::runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, stream_id, kNumber1,
                                                                     weight[i].tensor());
  }
  std::vector<ms::Tensor> new_weight;
  UnifyWeightShape(weight, &new_weight);
  if (weight_format == kFractalNzFormat) {
    for (auto &w_tensor : new_weight) {
      w_tensor.set_format(weight_format);
      auto storage_info = GetNZFormatStorageInfo(w_tensor.shape(), w_tensor.data_type());
      MS_EXCEPTION_IF_NULL(w_tensor.tensor());
      MS_EXCEPTION_IF_NULL(w_tensor.tensor()->device_address());
      w_tensor.tensor()->device_address()->set_tensor_storage_info(storage_info);
    }
  }

  auto bias_tensor = GetOptionalTensorList(bias);
  auto scale_tensor = GetOptionalTensorList(scale);
  auto offset_tensor = GetOptionalTensorList(offset);
  auto antiquant_scale_tensor = GetOptionalTensorList(antiquant_scale);
  auto antiquant_offset_tensor = GetOptionalTensorList(antiquant_offset);
  auto pre_token_scale_tensor = GetOptionalTensorList(per_token_scale);
  auto activation_input_tensor = GetOptionalTensorList(activation_input);
  auto activation_quant_scale_tensor = GetOptionalTensorList(activation_quant_scale);
  auto activation_quant_offset_tensor = GetOptionalTensorList(activation_quant_offset);

  std::vector<ms::Tensor> inputs;
  GetFlattenInputs(x, inputs);
  GetFlattenInputs(weight, inputs);
  GetFlattenInputs(bias_tensor, inputs);
  GetFlattenInputs(scale_tensor, inputs);
  GetFlattenInputs(offset_tensor, inputs);
  GetFlattenInputs(antiquant_scale_tensor, inputs);
  GetFlattenInputs(antiquant_offset_tensor, inputs);
  GetFlattenInputs(pre_token_scale_tensor, inputs);
  (void)inputs.emplace_back(GetTensorOrEmpty(group_list));
  GetFlattenInputs(activation_input_tensor, inputs);
  GetFlattenInputs(activation_quant_scale_tensor, inputs);
  GetFlattenInputs(activation_quant_offset_tensor, inputs);

  std::vector<ms::Tensor> activation_feature_out;
  std::vector<ms::Tensor> dyn_quant_scale_out;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulV4");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulV4, x, new_weight, bias_tensor, scale_tensor, offset_tensor,
                                          antiquant_scale_tensor, antiquant_offset_tensor, pre_token_scale_tensor,
                                          group_list, activation_input_tensor, activation_quant_scale_tensor,
                                          activation_quant_offset_tensor, split_item, group_type, group_list_type,
                                          act_type, outputs, activation_feature_out, dyn_quant_scale_out));
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_grouped_matmul_v4_cops(
  const std::vector<ms::Tensor> &x, const std::vector<ms::Tensor> &weight,
  const std::optional<std::vector<ms::Tensor>> &bias, const std::optional<std::vector<ms::Tensor>> &scale,
  const std::optional<std::vector<ms::Tensor>> &offset, const std::optional<std::vector<ms::Tensor>> &antiquant_scale,
  const std::optional<std::vector<ms::Tensor>> &antiquant_offset,
  const std::optional<std::vector<ms::Tensor>> &per_token_scale, const std::optional<ms::Tensor> &group_list,
  const std::optional<std::vector<ms::Tensor>> &activation_input,
  const std::optional<std::vector<ms::Tensor>> &activation_quant_scale,
  const std::optional<std::vector<ms::Tensor>> &activation_quant_offset, const int64_t &split_item,
  const int64_t &group_type, const int64_t &group_list_type, const int64_t &act_type, const std::string &weight_format,
  const std::optional<int64_t> &output_dtype) {
  {
    // grouped_matmul_v4 does not support asynchronous execution due to an uncertain number of outputs. To ensure proper
    // sequencing, it is necessary to wait for the frontend queue to clear first.
    ms_custom_ops::GilReleaseWithCheck no_gil;
    mindspore::runtime::Pipeline::Get().frontend_stage()->Wait();
  }
  ms::inner::ConvertStubNodeToTensor(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale,
                                     group_list, activation_input, activation_quant_scale, activation_quant_offset);
  std::vector<ms::Tensor> outputs = ms_custom_ops::grouped_matmul_v4_cops_custom(
    x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input,
    activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, weight_format,
    output_dtype);
  return outputs;
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("grouped_matmul_v4_cops", &pyboost_grouped_matmul_v4_cops, "GroupedMatmulV4", pybind11::arg("x"),
        pybind11::arg("weight"), pybind11::arg("bias") = std::nullopt, pybind11::arg("scale") = std::nullopt,
        pybind11::arg("offset") = std::nullopt, pybind11::arg("antiquant_scale") = std::nullopt,
        pybind11::arg("antiquant_offset") = std::nullopt, pybind11::arg("per_token_scale") = std::nullopt,
        pybind11::arg("group_list") = std::nullopt, pybind11::arg("activation_input") = std::nullopt,
        pybind11::arg("activation_quant_scale") = std::nullopt, pybind11::arg("activation_quant_offset") = std::nullopt,
        pybind11::arg("split_item") = 0, pybind11::arg("group_type") = -1, pybind11::arg("group_list_type") = 0,
        pybind11::arg("act_type") = 0, pybind11::arg("weight_format") = std::string("ND"),
        pybind11::arg("output_dtype") = std::nullopt);
}
