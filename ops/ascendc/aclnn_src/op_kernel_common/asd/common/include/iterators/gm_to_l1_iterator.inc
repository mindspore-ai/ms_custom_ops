/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "../iterator.h"

constexpr uint32_t STRIDE_LIMIT = 65536;

// Partial specialization for V220, ND_in, ND_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::ND> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        AscendC::DataCopy(l1Tensor,
                          gmTensor,
                          AscendC::DataCopyParams(1,                                              // nBurst
                                                  CeilDiv<BLOCK_SIZE>(nTileActual * dTileActual), // lenBurst
                                                  0,                                              // srcGap
                                                  0));                                            // dstGap
    };
};

// Partial specialization for NZ_in, NZ_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::NZ, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        uint64_t srcStride = nTileCeil - nTileActual;
        if (srcStride < STRIDE_LIMIT) {
            AscendC::DataCopy(l1Tensor, gmTensor,
                              AscendC::DataCopyParams(dTileActual / BLOCK_SIZE, // nBurst
                                                      nTileActual,              // lenBurst
                                                      nTileCeil - nTileActual,  // srcGap
                                                      0));                      // dstGap
        } else {
            for (uint64_t i = 0; i < dTileActual / BLOCK_SIZE; i++) {
                uint64_t dstOffset = i * nTileActual * BLOCK_SIZE;
                uint64_t srcOffset = i * nTileCeil * BLOCK_SIZE;
                AscendC::DataCopy(l1Tensor[dstOffset], gmTensor[srcOffset],
                                  AscendC::DataCopyParams(1,           // nBurst
                                                          nTileActual, // lenBurst
                                                          0,           // srcGap
                                                          0));         // dstGap
            }
        }
    };
};

// Partial specialization for V220, ND_in, ND_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        if (dVal < STRIDE_LIMIT) {
            AscendC::DataCopy(l1Tensor,
                              gmTensor,
                              AscendC::Nd2NzParams(1,           // ndNum
                                                   nTileActual, // nValue
                                                   dTileActual, // dValue
                                                   0,           // srcNdMatrixStride, unused
                                                   dVal,        // srcDValue
                                                   nTileCeil,   // dstNzC0Stride
                                                   1,           // dstNzNStride
                                                   0));         // dstNzMatrixStride, unused
        } else {
            for (uint32_t i = 0; i < nTileActual; i++) {
                AscendC::DataCopy(l1Tensor[i * BLOCK_SIZE],
                                  gmTensor[i * dVal],
                                  AscendC::Nd2NzParams(1,           // ndNum
                                                       1,           // nValue
                                                       dTileActual, // dValue
                                                       0,           // srcNdMatrixStride, unused
                                                       0,           // srcDValue
                                                       nTileCeil,   // dstNzC0Stride
                                                       0,           // dstNzNStride
                                                       0));         // dstNzMatrixStride, unused
            }
        }
    };
};

// Partial specialization for V220, ND_in, NZ_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::ZN> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        if (dVal < STRIDE_LIMIT) {
            AscendC::DataCopy(l1Tensor,
                              gmTensor,
                              AscendC::Nd2NzParams(1,           // ndNum
                                                   nTileActual, // nValue
                                                   dTileActual, // dValue
                                                   0,           // srcNdMatrixStride, unused
                                                   dVal,        // srcDValue
                                                   nTileCeil,   // dstNzC0Stride
                                                   1,           // dstNzNStride
                                                   0));         // dstNzMatrixStride, unused
        } else {
            for (uint32_t i = 0; i < nTileActual; ++i) {
                AscendC::DataCopy(l1Tensor,
                                  gmTensor,
                                  AscendC::Nd2NzParams(1,           // ndNum
                                                       1,           // nValue
                                                       dTileActual, // dValue
                                                       0,           // srcNdMatrixStride, unused
                                                       0,           // srcDValue
                                                       nTileCeil,   // dstNzC0Stride
                                                       0,           // dstNzNStride
                                                       0));         // dstNzMatrixStride, unused
            }
        }
    };
};

// Partial specialization for V220, ND_in, NZ_out
template <ArchType ArchTag, typename DataType>
struct m_gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::NZ> {
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

  __aicore__ m_gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor, AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t ndNum, uint32_t nValue, uint32_t dValue, uint32_t srcNdMatrixStride,
                        uint32_t srcDValue, uint32_t dstNzC0Stride, uint32_t dstNzNStride, uint32_t dstNzMatrixStride) {
    if (srcNdMatrixStride < STRIDE_LIMIT) {
      AscendC::DataCopy(l1Tensor, gmTensor,
                        AscendC::Nd2NzParams(ndNum,                // ndNum
                                             nValue,               // nValue
                                             dValue,               // dValue
                                             srcNdMatrixStride,    // srcNdMatrixStride, unused
                                             srcDValue,            // srcDValue
                                             dstNzC0Stride,        // dstNzC0Stride
                                             dstNzNStride,         // dstNzNStride
                                             dstNzMatrixStride));  // dstNzMatrixStride, unused
    } else {
        for (int i = 0; i < ndNum; i++) {
            AscendC::DataCopy(l1Tensor[i * dstNzMatrixStride], gmTensor[i * srcNdMatrixStride],
                AscendC::Nd2NzParams(1,                // ndNum
                                     nValue,               // nValue
                                     dValue,               // dValue
                                     0,                    // unused
                                     srcDValue,            // srcDValue
                                     dstNzC0Stride,        // dstNzC0Stride
                                     dstNzNStride,         // dstNzNStride
                                     0));                  // unused            
        }
    }
  };
};

template <ArchType ArchTag, typename DataType>
  __aicore__ __attribute__((always_inline)) inline void CopyGmToL1Npd(AscendC::LocalTensor<DataType> l1_dst,
                                                                            AscendC::GlobalTensor<DataType> gm_src,
                                                                            size_t page_size, size_t kv_len,
                                                                            size_t kv_len_round, size_t embed,
                                                                            size_t embed_round, size_t kv_stride) {
    
    
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType); 
    uint32_t nd_num = kv_len / page_size;
    if (nd_num > 0) {
      m_gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::NZ>(
        l1_dst, gm_src,
        nd_num,                   // ndNum
        page_size,                // nValue
        embed,                    // dValue
        kv_stride,                // srcNdMatrixStride
        embed_round,              // srcDValue
        kv_len_round,             // dstNzC0Stride
        1,                        // dstNzNStride
        page_size * BLOCK_SIZE);  // dstNzMatrixStride
    }

    uint32_t reminder = kv_len % page_size;
    if (reminder > 0) {
      size_t l1_k_offset = (nd_num) * page_size * BLOCK_SIZE;
      size_t gm_k_offset = (nd_num) * kv_stride;

      m_gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::NZ>(
        l1_dst[l1_k_offset], gm_src[gm_k_offset],
        1,                        // ndNum
        reminder,                 // nValue
        embed,                    // dValue
        kv_stride,            // srcNdMatrixStride
        embed_round,              // srcDValues
        kv_len_round,             // dstNzC0Stride
        1,                        // dstNzNStride
        page_size * BLOCK_SIZE);  // dstNzMatrixStride
    }
    return;
  };