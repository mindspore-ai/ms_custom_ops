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
#include <set>
#include <memory>
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"
#include "ops/c_api/utils/check_utils.h"

namespace ms_custom_ops {
enum class KvRmsNormRopeCacheInputIndex : size_t {
  kKvRmsNormRopeCacheKvIndex = 0,
  kKvRmsNormRopeCacheGammaIndex,
  kKvRmsNormRopeCacheCosIndex,
  kKvRmsNormRopeCacheSinIndex,
  kKvRmsNormRopeCacheIdxIndex,
  kKvRmsNormRopeCacheKCacheIndex,
  kKvRmsNormRopeCacheCKvCacheIndex,
  kKvRmsNormRopeCacheKRopeScaleIndex,
  kKvRmsNormRopeCacheCKvScaleIndex,
  kKvRmsNormRopeCacheKRopeOffsetIndex,
  kKvRmsNormRopeCacheCKvOffsetIndex,
  kKvRmsNormRopeCacheEpsilonIndex,
  kKvRmsNormRopeCacheCacheModeIndex,
  kKvRmsNormRopeCacheIsOutputKvIndex,
  kKvRmsNormRopeCacheInputNums,
};

enum class KvRmsNormRopeCacheSize : size_t {
  kKvRmsNormRopeCacheSize0 = 0,
  kKvRmsNormRopeCacheSize1,
  kKvRmsNormRopeCacheSize2,
  kKvRmsNormRopeCacheSize3,
  kKvRmsNormRopeCacheSize4,
  kKvRmsNormRopeCacheSize5,
};

enum class KvRmsNormRopeCacheModeOutput : size_t {
  kKvRmsNormRopeCacheKRopeOutIndex = 0,
  kKvRmsNormRopeCacheCKvOutIndex,
  kKvRmsNormRopeCacheOutNums,
};

enum class KvRmsNormRopeCacheMode : size_t {
  kKvRmsNormRopeCacheModeNorm = 0,
  kKvRmsNormRopeCacheModePA,
  kKvRmsNormRopeCacheModePA_BNSD,
  kKvRmsNormRopeCacheModePA_NZ,
  kKvRmsNormRopeCacheModePA_BLK_BNSD,
  kKvRmsNormRopeCacheModePA_BLK_NZ,
};

static std::map<KvRmsNormRopeCacheMode, std::string> KvRmsNormRopeCacheModeMap = {
  {KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModeNorm, "Norm"},
  {KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModePA, "PA"},
  {KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModePA_BNSD, "PA_BNSD"},
  {KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModePA_NZ, "PA_NZ"},
  {KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModePA_BLK_BNSD, "PA_BLK_BNSD"},
  {KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModePA_BLK_NZ, "PA_BLK_NZ"},
};

static std::string get_kv_rmsnorm_rope_cache_mode(KvRmsNormRopeCacheMode cache_mode) {
  auto it = KvRmsNormRopeCacheModeMap.find(cache_mode);
  if (it == KvRmsNormRopeCacheModeMap.end()) {
    MS_EXCEPTION(ValueError)
      << "For kv_rmsnorm_rope_cache, the cache mode should be Norm/PA/PA_BNSD/PA_NZ/PA_BLK_BNSD/PA_BLK_NZ, but got:"
      << cache_mode;
  }
  return it->second;
}

class OPS_API KvRmsNormRopeCacheCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto kv_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKvIndex)]->GetShape();
    auto k_cache_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKCacheIndex)]->GetShape();
    auto c_kv_cache_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCKvCacheIndex)]->GetShape();
    auto is_output_kv =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheIsOutputKvIndex)]
        ->GetScalarValueWithCheck<bool>();
    ShapeVector empty_shape;
    if ((input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKvIndex)]->IsDynamicRank()) ||
        (input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKCacheIndex)]
           ->IsDynamicRank()) ||
        (input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCKvCacheIndex)]
           ->IsDynamicRank())) {
      auto output_shape = ShapeVector{abstract::Shape::kShapeRankAny};
      if (is_output_kv) {
        return {output_shape, output_shape};
      } else {
        return {empty_shape, empty_shape};
      }
    }

    auto gamma_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheGammaIndex)]->GetShape();
    auto cos_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCosIndex)]->GetShape();
    auto sin_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheSinIndex)]->GetShape();
    auto dv = gamma_shape.back();
    auto dk = cos_shape.back();
    auto k_rope_shape = kv_shape;
    auto c_kv_out_shape = kv_shape;
    k_rope_shape[k_rope_shape.size() - 1] = dk;
    c_kv_out_shape[c_kv_out_shape.size() - 1] = dv;

    // dynamic shape or static shape;
    MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
      (kv_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4)), op_name,
      " kv input size should be " +
        std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4)) + ", but now got " +
        std::to_string(kv_shape.size()));

    MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
      (gamma_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize1)), op_name,
      " gamma input size should be " +
        std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize1)) + ", but now got " +
        std::to_string(gamma_shape.size()));

    MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
      ((cos_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4))), op_name,
      std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4)) + ", but now got cos " +
        std::to_string(cos_shape.size()));

    MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
      ((sin_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4))), op_name,
      std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4)) + ", but now got sin " +
        std::to_string(sin_shape.size()));

    auto cache_mode = static_cast<KvRmsNormRopeCacheMode>(
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCacheModeIndex)]
        ->GetScalarValueWithCheck<int64_t>());
    auto index_shape =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheIdxIndex)]->GetShape();
    if (cache_mode == KvRmsNormRopeCacheMode::kKvRmsNormRopeCacheModeNorm) {
      MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
        (index_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize2)), op_name,
        " index input size should be " +
          std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize2)) + ", but now got" +
          std::to_string(index_shape.size()));
    } else {
      MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
        (index_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize1)), op_name,
        " index input size should be " +
          std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize1)) + ", but now got" +
          std::to_string(index_shape.size()));
    }

    MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(
      ((k_cache_shape.size() == static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4)) ||
       (c_kv_cache_shape.size() != static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4))),
      op_name,
      " kCache or CKvCache input size should be " +
        std::to_string(static_cast<size_t>(KvRmsNormRopeCacheSize::kKvRmsNormRopeCacheSize4)) +
        ", but now got k_cache:" + std::to_string(k_cache_shape.size()) +
        ", c_kv_cache:" + std::to_string(c_kv_cache_shape.size()));
    if (is_output_kv) {
      return {k_rope_shape, c_kv_out_shape};
    } else {
      return {empty_shape, empty_shape};
    }
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto kv_dtype =
      input_infos[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKvIndex)]->GetType();
    const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
    CheckAndConvertUtils::CheckTypeIdValid("kv", kv_dtype, valid_types, op_name);
    return {kv_dtype, kv_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class KvRmsNormRopeCacheCustomAscend : public AclnnCustomKernelMod {
 public:
  KvRmsNormRopeCacheCustomAscend() : AclnnCustomKernelMod("aclnnKvRmsNormRopeCache") {}
  ~KvRmsNormRopeCacheCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKvIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheGammaIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCosIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheSinIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheIdxIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKCacheIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCKvCacheIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKRopeScaleIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCKvScaleIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKRopeOffsetIndex)],
          inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKRopeOffsetIndex)], epsilon_,
          cache_mode_str_, is_output_kv_,
          outputs[static_cast<size_t>(KvRmsNormRopeCacheModeOutput::kKvRmsNormRopeCacheKRopeOutIndex)],
          outputs[static_cast<size_t>(KvRmsNormRopeCacheModeOutput::kKvRmsNormRopeCacheCKvOutIndex)]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    cache_mode_ = device::ascend::ConvertKernelTensor<int64_t>(
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCacheModeIndex)]);
    cache_mode_str_ = get_kv_rmsnorm_rope_cache_mode(static_cast<KvRmsNormRopeCacheMode>(cache_mode_));
    epsilon_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheEpsilonIndex)]));
    is_output_kv_ = device::ascend::ConvertKernelTensor<bool>(
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheIsOutputKvIndex)]);
    GetWorkspaceForResize(
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKvIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheGammaIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCosIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheSinIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheIdxIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKCacheIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCKvCacheIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKRopeScaleIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheCKvScaleIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKRopeOffsetIndex)],
      inputs[static_cast<size_t>(KvRmsNormRopeCacheInputIndex::kKvRmsNormRopeCacheKRopeOffsetIndex)], epsilon_,
      cache_mode_str_, is_output_kv_,
      outputs[static_cast<size_t>(KvRmsNormRopeCacheModeOutput::kKvRmsNormRopeCacheKRopeOutIndex)],
      outputs[static_cast<size_t>(KvRmsNormRopeCacheModeOutput::kKvRmsNormRopeCacheCKvOutIndex)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  double epsilon_{1e-5};
  int64_t cache_mode_{0};
  bool is_output_kv_{false};
  std::string cache_mode_str_{"Norm"};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(kv_rmsnorm_rope_cache, ms_custom_ops::KvRmsNormRopeCacheCustomOpFuncImpl,
                  ms_custom_ops::KvRmsNormRopeCacheCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
std::vector<ms::Tensor> kv_rmsnorm_rope_cache_custom(
  const ms::Tensor &kv, const ms::Tensor &gamma, const ms::Tensor &cos, const ms::Tensor &sin, const ms::Tensor &index,
  const ms::Tensor k_cache, const ms::Tensor &c_kv_cache, const std::optional<ms::Tensor> &k_rope_scale,
  const std::optional<ms::Tensor> &c_kv_scale, const std::optional<ms::Tensor> &k_rope_offset,
  const std::optional<ms::Tensor> &c_kv_offset, const float epsilon, const int64_t cache_mode,
  const bool is_output_kv) {
  auto kv_shape = kv.shape();
  auto dv = gamma.shape().back();
  auto dk = cos.shape().back();
  ShapeVector k_rope_shape;
  ShapeVector c_kv_out_shape;

  if (is_output_kv) {
    k_rope_shape = kv_shape;
    c_kv_out_shape = kv_shape;
    k_rope_shape[k_rope_shape.size() - 1] = dk;
    c_kv_out_shape[c_kv_out_shape.size() - 1] = dv;
  }

  std::vector<ms::Tensor> outputs = {
    ms::Tensor(kv.data_type(), k_rope_shape),
    ms::Tensor(kv.data_type(), c_kv_out_shape),
  };
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnKvRmsNormRopeCache");
  auto cache_mode_str = get_kv_rmsnorm_rope_cache_mode(static_cast<KvRmsNormRopeCacheMode>(cache_mode));
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
    aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, c_kv_cache, k_rope_scale, c_kv_scale, k_rope_offset,
    c_kv_offset, static_cast<double>(epsilon), cache_mode_str, is_output_kv, outputs[0], outputs[1]));
  // only set tensor.
  runner->Run(
    {
      kv,
      gamma,
      cos,
      sin,
      index,
      k_cache,
      c_kv_cache,
      GetTensorOrEmpty(k_rope_scale),
      GetTensorOrEmpty(c_kv_scale),
      GetTensorOrEmpty(k_rope_offset),
      GetTensorOrEmpty(c_kv_offset),
    },
    outputs);
  return {outputs[0], outputs[1]};
}  // namespace ms_custom_ops
}  // namespace ms_custom_ops

auto pyboost_kv_rmsnorm_rope_cache(const ms::Tensor &kv, const ms::Tensor &gamma, const ms::Tensor &cos,
                                   const ms::Tensor &sin, const ms::Tensor &index, const ms::Tensor k_cache,
                                   const ms::Tensor &c_kv_cache, const std::optional<ms::Tensor> &k_rope_scale,
                                   const std::optional<ms::Tensor> &c_kv_scale,
                                   const std::optional<ms::Tensor> &k_rope_offset,
                                   const std::optional<ms::Tensor> &c_kv_offset, const float epsilon,
                                   const int64_t cache_mode, const bool is_output_kv) {
  return ms::pynative::PyboostRunner::Call<static_cast<size_t>(
    ms_custom_ops::KvRmsNormRopeCacheModeOutput::kKvRmsNormRopeCacheOutNums)>(
    ms_custom_ops::kv_rmsnorm_rope_cache_custom, kv, gamma, cos, sin, index, k_cache, c_kv_cache, k_rope_scale,
    c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode, is_output_kv);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("kv_rmsnorm_rope_cache", &pyboost_kv_rmsnorm_rope_cache, "KV Rmsnorm Rope Cache", pybind11::arg("kv"),
        pybind11::arg("gamma"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("index"),
        pybind11::arg("k_cache"), pybind11::arg("c_kv_cache"), pybind11::arg("k_rope_scale") = std::nullopt,
        pybind11::arg("c_kv_scale") = std::nullopt, pybind11::arg("k_rope_offset") = std::nullopt,
        pybind11::arg("c_kv_offset") = std::nullopt, pybind11::arg("epsilon") = 1e-5, pybind11::arg("cache_mode") = 0,
        pybind11::arg("is_output_kv") = false);
}
