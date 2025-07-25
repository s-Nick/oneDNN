/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/gpu_impl_list.hpp"

#include "gpu/generic/cross_engine_reorder.hpp"
#include "gpu/generic/direct_copy.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/custom_reorder.hpp"
#include "gpu/intel/generic_reorder.hpp"
#include "gpu/intel/jit/reorder/gen_reorder.hpp"
#include "gpu/intel/ref_reorder.hpp"
#include "gpu/intel/rnn/reorders.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_reorder.hpp"
#include "gpu/nvidia/cudnn_reorder_lt.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#include "gpu/amd/miopen_reorder.hpp"
#endif

#ifdef GENERIC_SYCL_KERNELS_ENABLED
#include "gpu/generic/sycl/ref_reorder.hpp"
#endif
namespace dnnl {
namespace impl {
namespace gpu {

namespace {

using namespace dnnl::impl::data_type;

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_REORDER_P({
        GPU_REORDER_INSTANCE_INTEL(intel::rnn_weights_reorder_t::pd_t)
        GPU_REORDER_INSTANCE_GENERIC(generic::direct_copy_t::pd_t)
        GPU_REORDER_INSTANCE_INTEL(intel::jit::gen_reorder_t::pd_t)
        GPU_REORDER_INSTANCE_INTEL(intel::custom_reorder_t::pd_t) // for specific tensor shapes
        GPU_REORDER_INSTANCE_INTEL(intel::generic_reorder_t::pd_t)// fast and quite generic
        GPU_REORDER_INSTANCE_INTEL(intel::ref_reorder_t::pd_t)    // slow but fits every use case
        GPU_REORDER_INSTANCE_NVIDIA(nvidia::cudnn_reorder_lt_t::pd_t)
        GPU_REORDER_INSTANCE_NVIDIA(nvidia::cudnn_reorder_t::pd_t)
        GPU_REORDER_INSTANCE_AMD(amd::miopen_reorder_t::pd_t)
        GPU_REORDER_INSTANCE_GENERIC(generic::cross_engine_reorder_t::pd_t)
        GPU_REORDER_INSTANCE_GENERIC_SYCL(generic::sycl::ref_reorder_t::pd_t)
        nullptr,
});
// clang-format on

} // namespace

const impl_list_item_t *get_reorder_impl_list(
        const memory_desc_t *, const memory_desc_t *) {
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
