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

#include "ascendc/basic_api/kernel_operator.h"

#ifdef __DAV_C220_VEC__
static constexpr int32_t BLOCK_SIZE = 32;
static constexpr uint16_t P1_P2_SYNC_FLAG = 1;
template <bool kv_cache_npd, bool kv_npd, typename Dtype>
class ReshapeAndCache {
 public:
  void __aicore__ __inline__ Init(GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                  GM_ADDR slot_mapping, GM_ADDR q_seq, GM_ADDR kv_seq, GM_ADDR block_tbl, GM_ADDR k_out,
                                  GM_ADDR v_out, GM_ADDR workspace, GM_ADDR tiling) {
    in_key_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(key));
    in_value_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(value));
    in_out_key_cache_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(key_cache));
    in_out_value_cache_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(value_cache));
    in_slot_mapping_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slot_mapping));
    in_block_tbl_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_tbl));
    out_key_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(k_out));
    out_value_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(v_out));
    in_tiling_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tiling));

    p1_core_num_ = in_tiling_.GetValue(1);
    num_tokens_ = in_tiling_.GetValue(2);
    hidden_size_ = in_tiling_.GetValue(3);
    k_head_num_ = in_tiling_.GetValue(4);
    v_head_num_ = in_tiling_.GetValue(5);
    page_size_ = in_tiling_.GetValue(6);
    batch_ = in_tiling_.GetValue(7);
    ub_size_ = in_tiling_.GetValue(8);
    ws_size_ = in_tiling_.GetValue(9);
    in_workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + ws_size_ * get_block_idx());
    blk_tbl_dim1_ = in_tiling_.GetValue(10);
    page_size_n_ = page_size_ * hidden_size_;
    local_buf_n_ = ub_size_ / 2 / sizeof(Dtype);
    in_local_ping_ = ub_allocator_.Alloc<Dtype>(local_buf_n_);
    in_local_pong_ = ub_allocator_.Alloc<Dtype>(local_buf_n_);
    local_buf_n_floor_ = local_buf_n_ / hidden_size_ * hidden_size_;  // size of buffer for full tokens
    local_buf_tokens_n_ = local_buf_n_floor_ / hidden_size_;          // number of tokens
    in_kv_seq_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling) + 32);
    in_q_seq_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling) + 32 + batch_);

    SetupPrefix();
  }

  void __aicore__ __inline__ Process() {
    int32_t block_id = get_block_idx();
    // trigger events
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);

    if (block_id < p1_core_num_) {
      int64_t per_core_task_num = num_tokens_ * 2 / p1_core_num_;
      int32_t tail_task_num = num_tokens_ * 2 % p1_core_num_;

      int64_t start_task_id = block_id * per_core_task_num;

      if (block_id < tail_task_num) {
        per_core_task_num++;
        start_task_id += block_id;
      } else {
        start_task_id += tail_task_num;
      }
      for (int64_t i = 0; i < per_core_task_num; i++) {
        if (i + start_task_id < num_tokens_) {
          ReshapeAndCacheDo(in_key_, k_head_num_, hidden_size_ / k_head_num_, (i + start_task_id), in_out_key_cache_,
                            out_key_);
        } else {
          ReshapeAndCacheDo(in_value_, v_head_num_, hidden_size_ / v_head_num_, (i + start_task_id - num_tokens_),
                            in_out_value_cache_, out_value_);
        }
      }
    }

    // copy from cache into out buffers
    test_cache_ = false;
    // make sure all cores are done with phase 1
    AscendC::SyncAll<true>();
    if (block_id < p2_core_num_) {
      // copy prefix into
      int64_t per_core_task_num = cpy_page_ * 2 / p2_core_num_;
      int32_t tail_task_num = cpy_page_ * 2 % p2_core_num_;
      int64_t start_task_id = block_id * per_core_task_num;

      if (block_id < tail_task_num) {
        per_core_task_num++;
        start_task_id += block_id;
      } else {
        start_task_id += tail_task_num;
      }
      for (int64_t i = 0; i < per_core_task_num; i++) {
        if (i + start_task_id < cpy_page_) {
          ReshapeAndCacheCpyPageDo((i + start_task_id), k_head_num_, in_out_key_cache_, out_key_);
        } else {
          ReshapeAndCacheCpyPageDo((i + start_task_id - cpy_page_), v_head_num_, in_out_value_cache_, out_value_);
        }
      }
    }
    // release events
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
  }

 private:
  bool __aicore__ __inline__ IsDecode(int b) { return in_q_seq_.GetValue(b) == 1; }

  void __aicore__ __inline__ SetupPrefix() {
    for (int i = 0; i < batch_; i++) {
      auto kv_seq = in_kv_seq_.GetValue(i);
      auto cur_cpy_page = (kv_seq + page_size_ - 1) / page_size_;
      cpy_page_ += cur_cpy_page;
    }

    p2_core_num_ = get_block_num();
    if (cpy_page_ < p2_core_num_) {
      p2_core_num_ = cpy_page_;
    }
  }

  void __aicore__ __inline__ CopyIn(AscendC::LocalTensor<Dtype> &in_local, AscendC::GlobalTensor<Dtype> &in_kv,
                                    uint64_t src_offset, AscendC::DataCopyParams cpy_param) {
    DataCopy(in_local, in_kv[src_offset], cpy_param);
  }

  void __aicore__ __inline__ CopyOut(AscendC::GlobalTensor<Dtype> &out, AscendC::LocalTensor<Dtype> &in_local,
                                     uint64_t dst_offset, AscendC::DataCopyParams cpy_param) {
    DataCopy(out[dst_offset], in_local, cpy_param);
  }

  void __aicore__ __inline__ ReshapeAndCacheDo(AscendC::GlobalTensor<Dtype> &in_kv, int32_t kv_head_num, int32_t embed,
                                               int32_t token_idx, AscendC::GlobalTensor<Dtype> &out_cache,
                                               AscendC::GlobalTensor<Dtype> &out_kv) {
    uint64_t start = token_idx * hidden_size_;
    int32_t slot_value = in_slot_mapping_.GetValue(token_idx);
    if (slot_value < 0) return;
    uint16_t num_hidden_blocks = hidden_size_ * sizeof(Dtype) / BLOCK_SIZE;
    uint16_t num_embed_blocks = embed * sizeof(Dtype) / BLOCK_SIZE;

    int32_t evt_id = EVENT_ID0 + token_idx % 2;
    AscendC::LocalTensor<Dtype> &in_local = (token_idx % 2) ? in_local_ping_ : in_local_pong_;
    // step I: copy token into local memory
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evt_id);  // local buffer is free to use
    AscendC::DataCopyParams cpy_in_param(1, num_hidden_blocks, 0, 0);
    CopyIn(in_local, in_kv, start, cpy_in_param);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evt_id);  // send event to sync when data is ready on local buffer
    // step II: copy token into kv_cache
    // cache is stored [B_NUM, B_SIZE, N, D]
    uint64_t cache_start = slot_value * hidden_size_;
    uint16_t dst_stride = 0;
    uint16_t n_burst = 1;
    uint16_t n_blks = num_hidden_blocks;
    if constexpr (kv_cache_npd) {
      // cache is stored [B_NUM, N, B_SIZE, D]
      cache_start = (slot_value / page_size_) * kv_head_num * page_size_ * embed + (slot_value % page_size_) * embed;
      dst_stride = (page_size_ - 1) * embed * sizeof(Dtype) / BLOCK_SIZE;
      n_burst = kv_head_num;
      n_blks = num_embed_blocks;
    }
    AscendC::DataCopyParams cpy_out_cache_param(n_burst, n_blks, 0, dst_stride);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evt_id);  // data is ready on local buffer
    CopyOut(out_cache, in_local, cache_start, cpy_out_cache_param);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evt_id);  // send event to sync when local buffer is available
    return;
  }

  bool __aicore__ __inline__ ComputePageIdCache(int32_t page_idx) {
    // check first if page idx is on current range
    if (test_cache_ && (page_idx >= range_st_ && page_idx <= range_ed_)) {
      return true;
    }
    return false;
  }

  void __aicore__ __inline__ ComputePageId(int32_t page_idx, int32_t *page_id, uint64_t *src, uint64_t *dst) {
    if (ComputePageIdCache(page_idx) == false) {
      range_st_ = 0;
      range_sum_round_ = 0;
      range_sum_round_skip_ = 0;
      range_sum_ = 0;
      for (int32_t i = 0; i < batch_; i++) {
        int32_t token_per_user = in_kv_seq_.GetValue(i);
        int32_t token_per_user_round = (token_per_user + page_size_ - 1) / page_size_;
        range_ed_ = range_st_ + token_per_user_round - 1;
        if (page_idx >= range_st_ && page_idx <= range_ed_) {
          range_id_ = i;
          test_cache_ = true;
          break;
        }
        range_st_ = range_ed_ + 1;
        range_sum_round_ += token_per_user_round;
        if (IsDecode(i)) range_sum_round_skip_ += token_per_user_round;
        range_sum_ += token_per_user;
      }
    }
    uint64_t inner_page_idx = page_idx - range_sum_round_;
    uint64_t batch_idx = range_id_;
    *page_id = -1;
    if (!IsDecode(batch_idx)) {
      *page_id = in_block_tbl_.GetValue(batch_idx * blk_tbl_dim1_ + inner_page_idx);
    }
    *src = *page_id * page_size_n_;
    *dst = (range_sum_round_ - range_sum_round_skip_ + inner_page_idx) * page_size_n_;
    return;
  }

  void __aicore__ __inline__ ReshapeAndCacheCpyOutPageNd2Npd(const AscendC::GlobalTensor<Dtype> &out_kv, int head_num,
                                                             int32_t tokens2cpy,
                                                             const AscendC::LocalTensor<Dtype> &in_local) {
    uint16_t embed = hidden_size_ / head_num;
    uint16_t num_embed_blocks = embed * sizeof(Dtype) / BLOCK_SIZE;

    uint16_t dst_stride = (page_size_ - 1) * embed * sizeof(Dtype) / BLOCK_SIZE;
    uint16_t n_burst = head_num;
    uint16_t n_blks = num_embed_blocks;
    AscendC::DataCopyParams cpy_out_param(n_burst, n_blks, 0, dst_stride);
    for (int i = 0; i < tokens2cpy; i++) {
      DataCopy(out_kv[i * embed], in_local[i * hidden_size_], cpy_out_param);
    }
  }

  void __aicore__ __inline__ ReshapeAndCacheCpyInPageNpd2Nd(const AscendC::GlobalTensor<Dtype> &in_ckv, int head_num,
                                                            int32_t tokens2cpy,
                                                            const AscendC::LocalTensor<Dtype> &in_local) {
    uint16_t embed = hidden_size_ / head_num;
    uint16_t num_embed_blocks = embed * sizeof(Dtype) / BLOCK_SIZE;

    uint16_t src_stride = (page_size_ - 1) * embed * sizeof(Dtype) / BLOCK_SIZE;
    uint16_t dst_stride = 0;
    uint16_t n_burst = head_num;
    uint16_t n_blks = num_embed_blocks;
    AscendC::DataCopyParams cpy_in_param(n_burst, n_blks, src_stride, dst_stride);
    for (int i = 0; i < tokens2cpy; i++) {
      DataCopy(in_local[i * hidden_size_], in_ckv[i * embed], cpy_in_param);
    }
  }

  void __aicore__ __inline__ ReshapeAndCacheCpyPageDo(int32_t page_idx, int32_t head_num,
                                                      AscendC::GlobalTensor<Dtype> &in_cache,
                                                      AscendC::GlobalTensor<Dtype> &out_kv) {
    int32_t page_id;
    uint64_t src, dst;
    ComputePageId(page_idx, &page_id, &src, &dst);
    if (page_id < 0) return;
    uint32_t embed = hidden_size_ / head_num;
    uint32_t copy_rounds = (page_size_ + local_buf_tokens_n_ - 1) / local_buf_tokens_n_;
    for (int32_t i = 0; i < copy_rounds; i++) {
      int32_t evt_id = EVENT_ID0 + (page_idx + i) % 2;
      AscendC::LocalTensor<Dtype> &in_local = ((page_idx + i) % 2) ? in_local_ping_ : in_local_pong_;
      int32_t cpy_size =
        ((i + 1) != copy_rounds) ? local_buf_n_floor_ : (page_size_ % local_buf_tokens_n_) * hidden_size_;
      // step I: copy token into local memory
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evt_id);  // local buffer is free to use
      if constexpr (kv_cache_npd == true && kv_npd == false) {
        ReshapeAndCacheCpyInPageNpd2Nd(in_cache[src + i * local_buf_tokens_n_ * embed], head_num,
                                       cpy_size / hidden_size_, in_local);
      } else {
        DataCopy(in_local, in_cache[src + i * local_buf_n_floor_], cpy_size);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evt_id);  // send event to sync when data is ready on local buffer
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evt_id);  // data is ready on local buffer

      if constexpr (kv_cache_npd == false && kv_npd == true) {
        ReshapeAndCacheCpyOutPageNd2Npd(out_kv[dst + i * local_buf_tokens_n_ * embed], head_num,
                                        cpy_size / hidden_size_, in_local);
      } else {
        DataCopy(out_kv[dst + i * local_buf_n_floor_], in_local, cpy_size);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evt_id);  // send event to sync when local buffer is available
    }
  }

  AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub_allocator_;
  AscendC::GlobalTensor<Dtype> in_key_;
  AscendC::GlobalTensor<Dtype> in_value_;
  AscendC::GlobalTensor<Dtype> in_out_key_cache_;
  AscendC::GlobalTensor<Dtype> in_out_value_cache_;
  AscendC::GlobalTensor<int32_t> in_slot_mapping_;
  AscendC::GlobalTensor<uint32_t> in_q_seq_;
  AscendC::GlobalTensor<uint32_t> in_kv_seq_;
  AscendC::GlobalTensor<int32_t> in_block_tbl_;
  AscendC::GlobalTensor<int32_t> in_tiling_;
  AscendC::GlobalTensor<int32_t> in_workspace_;
  AscendC::GlobalTensor<Dtype> out_key_;
  AscendC::GlobalTensor<Dtype> out_value_;
  AscendC::LocalTensor<Dtype> in_local_ping_;
  AscendC::LocalTensor<Dtype> in_local_pong_;

  int32_t p1_core_num_ = 0;
  int32_t p2_core_num_ = 0;
  int32_t num_tokens_ = 0;
  int32_t hidden_size_ = 0;
  int32_t k_head_num_ = 0;
  int32_t v_head_num_ = 0;
  int32_t page_size_ = 0;
  int32_t batch_ = 0;
  int32_t cpy_page_ = 0;
  int32_t ub_size_ = 0;
  int32_t range_st_ = 0;
  int32_t range_ed_ = 0;
  int32_t range_id_ = 0;
  int32_t range_sum_ = 0;
  int32_t range_sum_round_ = 0;
  int32_t range_sum_round_skip_ = 0;
  bool test_cache_ = false;
  int32_t page_size_n_ = 0;
  int32_t local_buf_n_floor_ = 0;
  int32_t local_buf_tokens_n_ = 0;
  int32_t local_buf_n_ = 0;
  int32_t blk_tbl_dim1_ = 0;
  int32_t ws_size_ = 0;
};

extern "C" __global__ __aicore__ void reshape_and_cache_npd(GM_ADDR key, GM_ADDR value, GM_ADDR key_cache,
                                                            GM_ADDR value_cache, GM_ADDR slot_mapping, GM_ADDR q_seq,
                                                            GM_ADDR kv_seq, GM_ADDR block_tbl, GM_ADDR k_out,
                                                            GM_ADDR v_out, GM_ADDR workspace, GM_ADDR tiling) {
  if (TILING_KEY_IS(0)) {
    ReshapeAndCache<false, false, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(1)) {
    ReshapeAndCache<true, false, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(2)) {
    ReshapeAndCache<false, true, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(3)) {
    ReshapeAndCache<true, true, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(4)) {
    ReshapeAndCache<false, false, int8_t> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(5)) {
    ReshapeAndCache<true, false, int8_t> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(6)) {
    ReshapeAndCache<false, true, int8_t> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(7)) {
    ReshapeAndCache<true, true, int8_t> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  }
}

#endif
