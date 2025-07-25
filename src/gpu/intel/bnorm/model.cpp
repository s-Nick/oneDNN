/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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
#include "gpu/intel/bnorm/model.hpp"
#include <cmath>
#include "common/utils.hpp"
#include "gpu/intel/bnorm/nhwc_batch_normalization.hpp"
#include "gpu/intel/bnorm/utils.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace bn_model {
using namespace dnnl::impl::utils;
using namespace dnnl::impl::gpu::intel::gpu_utils;
using namespace dnnl::impl::gpu::intel::bn_utils;

int get_nhwc_vect_size(int ic, int max_vect_size, int simd) {
    int vect_size = max_vect_size;
    while (true) {
        if (ic / (vect_size * simd)) return vect_size;
        vect_size /= 2;
    }
    return 1;
}
int get_nhwc_sp_block_size(
        dim_t sp, dim_t ic_dim, int eu_count, int threads_per_eu, int simd) {

    float efficiency_thr = 0.0f;
    float efficiency_peak_eu_thr = 0.0f;
    dim_t block_size_thr = 1;
    dim_t block_size_peak_eu_thr = 1;
    dim_t curr_block_size = sp;
    int nthr_mul = 1;
    const dim_t ic_nsg = ic_dim / simd; // number of subgroups by ic dim

    // The search is based on threads wave efficiency.
    // Higher priority for cases with peak EUs utilization.
    while (nthr_mul <= 32) {
        const int nthr = nthr_mul * eu_count;
        curr_block_size = div_up(sp * ic_nsg, nthr);
        const dim_t nblock = div_up(sp, curr_block_size);
        const dim_t nthr_gen = nblock * ic_nsg;

        const float curr_efficiency_eus
                = (float)nthr_gen / rnd_up(nthr_gen, eu_count);
        const float curr_efficiency_thr
                = (float)nthr_gen / rnd_up(nthr_gen, eu_count * threads_per_eu);

        if (curr_efficiency_thr > efficiency_thr) {
            efficiency_thr = curr_efficiency_thr;
            block_size_thr = curr_block_size;
        }
        if (curr_efficiency_eus == 1
                && curr_efficiency_thr > efficiency_peak_eu_thr) {
            efficiency_peak_eu_thr = curr_efficiency_thr;
            block_size_peak_eu_thr = curr_block_size;
        }
        nthr_mul++;
    }
    if (efficiency_peak_eu_thr > 0.0f) return into<int>(block_size_peak_eu_thr);
    return into<int>(block_size_thr);
}
dim_t get_nhwc_calc_stat_ic(dim_t ic, int ic_block, int sg_size) {
    return div_up(ic, ic_block) * sg_size;
}

void init_hw_params(hw_params_t &hw_params, impl::engine_t *engine) {
    const bool large_grf_mode = false;
    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    hw_params.gpu_arch = gpu_arch;
    hw_params.eu_count = compute_engine->device_info()->eu_count();
    hw_params.threads_per_eu
            = compute::device_info_t::threads_per_eu(gpu_arch, false);
    hw_params.max_lws
            = compute_engine->device_info()->max_wg_size(large_grf_mode);
    hw_params.eus_per_ss = compute_engine->device_info()->max_eus_per_wg();
    hw_params.max_ss = div_up(hw_params.eu_count, hw_params.eus_per_ss);
    hw_params.max_slm_size = compute::device_info_t::max_slm_size(gpu_arch);
    hw_params.engine = engine;

    // Experimentally selected, based on microbenchmarks results
    if (hw_params.gpu_arch == compute::gpu_arch_t::xe_hpg) {
        hw_params.HBM_bw = 400; //GBs
        hw_params.L3_size = 16 * (2 << 19); //Bytes
        hw_params.L3_bw = 2000; //GBs
        hw_params.host_overheads_per_kernel = 8000; // ns
    } else if (hw_params.gpu_arch >= compute::gpu_arch_t::xe_hpc) {
        hw_params.HBM_bw = 1000; //GBs
        hw_params.L3_size = 192 * (2 << 19); //Bytes
        hw_params.L3_bw = 3000; //GBs
        hw_params.host_overheads_per_kernel = 6000; // ns
    } else {
        assert(!"not supported");
    }
}

float get_used_ss_thr_utilization(hw_params_t &hw_params, int sg_size,
        const compute::range_t &gws, const compute::range_t &lws) {
    const size_t gws_size = gws.nelems();
    const size_t lws_size = lws.nelems();
    const size_t num_thrs_generated = gws_size / sg_size;
    const size_t num_wgs = gws_size / lws_size; // == ss used
    // TODO: considering case when several work groups are running
    // on the same [sub-]slice
    return (float)num_thrs_generated
            / std::min(
                    num_wgs * hw_params.eus_per_ss * hw_params.threads_per_eu,
                    into<size_t>(
                            hw_params.eu_count * hw_params.threads_per_eu));
}

std::string to_string(const kernel_kind_t &kernel) {
    std::string kernel_name;
    if (kernel == calc_mean_ker) {
        kernel_name = "calc_mean";
    } else if (kernel == calc_var_ker) {
        kernel_name = "calc_var";
    } else if (kernel == calc_mean_var_ker) {
        kernel_name = "calc_mean_var";
    } else if (kernel == calc_stats_ker) {
        kernel_name = "calc_stat";
    } else if (kernel == reduce_stats_fwd_ker) {
        kernel_name = "reduce_stats_fwd";
    } else if (kernel == reusable_reduce_stats_fwd_ker) {
        kernel_name = "reusable_reduce_stats_fwd";
    } else if (kernel == reduce_mean_var_ker) {
        kernel_name = "reduce_mean_var";
    } else if (kernel == reduce_stats_bwd_ker) {
        kernel_name = "reduce_stats_bwd";
    } else if (kernel == reduce_aux_init_ker) {
        kernel_name = "reduce_aux_init";
    } else if (kernel == reduce_aux_finalize_ker) {
        kernel_name = "reduce_aux_finalize";
    } else if (kernel == default_fwd_ker) {
        kernel_name = "default_fwd";
    } else if (kernel == default_bwd_ker) {
        kernel_name = "default_bwd";
    } else {
        gpu_error_not_expected();
    }
    return kernel_name;
}

std::string to_string(const data_location_t &loc) {
    std::string str_loc;
    if (loc == L3) {
        str_loc = "L3";
    } else if (loc == HBM) {
        str_loc = "HBM";
    } else if (loc == SLM) {
        str_loc = "SLM";
    } else {
        gpu_error_not_expected();
    }
    return str_loc;
}

// Useful for experimentation and debug purposes
void dump_kernel_descriptor(kernel_desc_t &desc) {
    DPRINT_MODEL(
            "%s:%s:%d kernel desc:  %s : ncalls = %d : nbytes = %lld %lld : "
            "location = %s %s\n",
            PRINTHEAD, to_string(desc.kernel).c_str(), desc.ncalls,
            into<long long>(desc.input_nbytes),
            into<long long>(desc.output_nbytes),
            to_string(desc.input_location).c_str(),
            to_string(desc.output_location).c_str());
}
std::string to_string(const nhwc_bnorm_params_t &conf) {
    std::string s;
#define STR_PARAM(p) \
    s += std::to_string(conf.p##_param().is_overridden()) + ","; \
    s += std::to_string((int)conf.p()) + ","

    STR_PARAM(use_fused_atomics_reduction);
    STR_PARAM(max_vect_size);
    s += std::to_string((int)conf.vect_size) + ",";
    STR_PARAM(ic_block);
    s += std::to_string((int)conf.sp) + ",";
    STR_PARAM(stat_sp_block);
    STR_PARAM(update_sp_block);
    STR_PARAM(update_sp_unroll);
    s += std::to_string((int)conf.sub_group_size) + ",";
    s += conf.found_in_table ? "LT" : std::to_string(conf.expected_time_ms);
    return s;
#undef STR_PARAM
}

// How short vector can increase r/w expected time
float get_vectorization_factor(
        const int vect_size, const data_type_t dt, const bool is_reusable) {
    const bool is_dt16 = dt == data_type::f16 || dt == data_type::bf16;
    if (is_reusable) {
        if (is_dt16) {
            switch (vect_size) {
                case 1: return 2.5f;
                case 2: return 1.8f;
                case 4: return 1.2f;
                case 8:
                default: return 1.f;
            }
        } else {
            switch (vect_size) {
                case 1: return 2.5f;
                case 2: return 1.5f;
                case 4:
                case 8:
                default: return 1.f;
            }
        }
    } else {
        if (is_dt16) {
            switch (vect_size) {
                case 1: return 4.f;
                case 2: return 1.5f;
                case 4: return 1.3f;
                case 8:
                default: return 1.f;
            }
        } else {
            switch (vect_size) {
                case 1: return 4.f;
                case 2: return 1.3f;
                case 4:
                case 8:
                default: return 1.f;
            }
        }
    }
}

// Get number of calls
int get_ncalls(model_params_t &p, const nhwc_bnorm_params_t &conf,
        kernel_kind_t kernel) {
    if (conf.is_forward) {
        switch (kernel) {
            case default_fwd_ker: return 1;
            case calc_mean_ker:
            case calc_var_ker:
            case calc_mean_var_ker: return conf.calculate_stats ? 1 : 0;
            case reusable_reduce_stats_fwd_ker:
            case reduce_stats_fwd_ker:
                return conf.calculate_stats && !p.use_fused_atomics_reduction
                        ? 2
                        : 0;
            case reduce_mean_var_ker:
                return conf.calculate_stats && !p.use_fused_atomics_reduction
                        ? 1
                        : 0;
            case reduce_aux_init_ker:
                return conf.calculate_stats && p.use_fused_atomics_reduction
                        ? 1
                        : 0;
            case reduce_aux_finalize_ker:
                return conf.calculate_stats && p.use_fused_atomics_reduction
                        ? (conf.use_stats_one_pass ? 1 : 2)
                        : 0;
            default: gpu_error_not_expected(); return 0;
        }
    } else { // BWD pass
        return 1;
    }
}

size_t get_kernel_input_size(const model_params_t &p,
        const nhwc_bnorm_params_t &conf, const kernel_desc_t &desc) {
    size_t nbytes = 0;
    const size_t tensor_sz = conf.sp * conf.ic * conf.elsz;
    const size_t stat_vect_sz = conf.ic * sizeof(float);
    const dim_t num_sp_blocks = div_up(conf.sp, p.stat_sp_block);
    const dim_t ws_sz = conf.sp * conf.ic * into<int>(sizeof(char));

    switch (desc.kernel) {
        case calc_mean_ker:
        case calc_mean_var_ker: nbytes = tensor_sz; break;
        case calc_var_ker:
            nbytes = tensor_sz + stat_vect_sz * num_sp_blocks;
            break;
        case reusable_reduce_stats_fwd_ker:
        case reduce_stats_fwd_ker:
            nbytes = num_sp_blocks * rnd_up(conf.ic, conf.sub_group_size)
                    * sizeof(float);
            break;
        case reduce_mean_var_ker:
            nbytes = 2 * num_sp_blocks * rnd_up(conf.ic, conf.sub_group_size)
                    * sizeof(float);
            break;
        case default_fwd_ker:
            nbytes = ((int)conf.fuse_norm_add_relu + 1) * tensor_sz
                    + ((int)conf.use_scale + (int)conf.use_shift + 2)
                            * stat_vect_sz;
            break;
        case reduce_aux_init_ker: break;
        case reduce_aux_finalize_ker:
            nbytes = stat_vect_sz
                    * (conf.is_backward ? 2
                                        : (conf.use_stats_one_pass ? 2 : 1));
            break;
        case default_bwd_ker:
            nbytes = 2 * tensor_sz
                    + (1 + (int)conf.calculate_diff_stats * 3
                              + (int)conf.use_scale)
                            * stat_vect_sz
                    + (int)conf.fuse_norm_relu * ws_sz;
            break;
        case calc_stats_ker:
            nbytes = 2 * tensor_sz + stat_vect_sz * num_sp_blocks
                    + (int)conf.fuse_norm_relu * ws_sz;
            break;
        case reduce_stats_bwd_ker:
            nbytes = 2 * num_sp_blocks * rnd_up(conf.ic, conf.sub_group_size)
                    * sizeof(float);
            break;

        default: gpu_error_not_expected();
    }
    return nbytes;
}
size_t get_kernel_output_size(const model_params_t &p,
        const nhwc_bnorm_params_t &conf, const kernel_desc_t &desc) {
    size_t nbytes = 0;
    const size_t tensor_sz = conf.sp * conf.ic * conf.elsz;
    const size_t stat_vect_sz = conf.ic * sizeof(float);
    const dim_t num_sp_blocks = div_up(conf.sp, p.stat_sp_block);

    switch (desc.kernel) {
        case calc_mean_ker:
        case calc_var_ker:
            nbytes = p.use_fused_atomics_reduction
                    ? stat_vect_sz * desc.num_wgs
                    : num_sp_blocks * rnd_up(conf.ic, conf.sub_group_size)
                            * sizeof(float);
            break;
        case calc_mean_var_ker:
            nbytes = p.use_fused_atomics_reduction
                    ? 2 * stat_vect_sz * desc.num_wgs
                    : 2 * num_sp_blocks * rnd_up(conf.ic, conf.sub_group_size)
                            * sizeof(float);
            break;
        case reduce_aux_init_ker: nbytes = 2 * stat_vect_sz; break;
        case reusable_reduce_stats_fwd_ker:
        case reduce_stats_fwd_ker: nbytes = stat_vect_sz; break;
        case reduce_mean_var_ker: nbytes = 2 * stat_vect_sz; break;
        case reduce_aux_finalize_ker:
            nbytes = stat_vect_sz
                    * (conf.is_forward && conf.use_stats_one_pass ? 2 : 1);
            break;
        case default_fwd_ker: nbytes = tensor_sz; break;
        case default_bwd_ker:
            nbytes = (1 + conf.fuse_norm_add_relu) * tensor_sz;
            break;
        case calc_stats_ker:
            nbytes = p.use_fused_atomics_reduction
                    ? 2 * stat_vect_sz * desc.num_wgs
                    : 2 * num_sp_blocks * rnd_up(conf.ic, conf.sub_group_size)
                            * sizeof(float);
            break;
        case reduce_stats_bwd_ker: nbytes = 2 * stat_vect_sz; break;
        default: gpu_error_not_expected();
    }
    return nbytes;
}
// Expected data location depending on arch, size and kernel kind.
void get_expected_data_location(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc) {
    desc.input_location = HBM;
    desc.output_location = HBM;

    // HBM only for XeHPG
    if (hw_params.gpu_arch == compute::gpu_arch_t::xe_hpg) return;

    if (desc.kernel == calc_mean_ker || desc.kernel == calc_var_ker) {
        if (desc.input_nbytes + desc.output_nbytes < hw_params.L3_size) {
            desc.input_location = L3;
        }
    } else if ((desc.kernel == default_fwd_ker && !conf.calculate_stats)
            || (desc.kernel == default_bwd_ker && !conf.calculate_diff_stats)) {
        // default kernels w/o stats calculation
        desc.input_location = HBM;
    } else { // all other kernels
        if (desc.input_nbytes < hw_params.L3_size) { desc.input_location = L3; }
    }
    if (desc.output_nbytes < hw_params.L3_size) { desc.output_location = L3; }
}

// linear approximation
// return y by x on the line passing thru (xa,ya) and (xb,yb)
float solve_2p_line(float x, float xa, float xb, float ya, float yb) {
    float dx = xb - xa;
    float dy = yb - ya;
    assert(dx != 0.0);
    return (dy / dx) * (x - xa) + ya;
}

// approximation by 2 pieces linear function
float solve_2pieces_linear_function(
        float x, float x0, float x1, float x2, float y0, float y1, float y2) {
    float y;
    if (x < x1) {
        y = solve_2p_line(x, x0, x1, y0, y1);
    } else {
        y = solve_2p_line(x, x1, x2, y1, y2);
    }
    return y;
}
float get_pow_ratio(float x1, float x2, float a, float b) {
    return a * pow(x1, b) / pow(x2, b);
}

// Subslice saturation impact to read/write time for all archs and data location.
float get_ss_utilization_factor(float util, data_type_t dt, bool is_reusable) {
    if (is_reusable) {
        if (dt == data_type::f16 || dt == data_type::bf16) {
            return get_pow_ratio(util, 1.0f, 2.0f, -0.8f);
        }
        return get_pow_ratio(util, 1.0f, 5.3f, -0.7f);
    } else {
        return 1.f / std::min(util, 1.f);
    }
}

std::vector<float> ss_util_set = {0.125, 0.25, 1, 2, 4, 8};
// clang-format off
appr_formula_t appr_table[] =
{
    //HBM reusable
    { 191, 0.27f, linear }, { 347, 5.03f, linear }, { 156, 595, ln },
    { 190, 785, ln }, { 165, 793, ln }, { 170, 926, ln },
    { 101, 0.02f, linear }, { 192, 1.5f, linear }, { 592, 27, linear },
    { 147, 571, ln }, { 174, 737, ln }, { 167, 811, ln }, { 103, 394, ln },
    { 99, 434, ln }, { 116, 707, ln }, { 140, 969, ln }, { 82, 949, ln },
    { 82, 1128, ln }, { 60, 220, ln }, { 82, 324, ln }, { 196, 832, ln },
    { 186, 936, ln }, { 201, 1192, ln }, { 176, 1330, ln },
    //HBM opt
    { 83, 313, ln }, { 131, 509, ln }, { 146, 733, ln }, { 153, 871, ln },
    { 120, 816, ln }, { 129, 914, ln }, { 296, 7.6f, linear },
    { 94, 345, ln }, { 177, 731, ln }, { 154, 740, ln }, { 159, 857, ln },
    { 143, 907, ln }, { 93, 389, ln }, { 125, 568, ln }, { 125, 820, ln },
    { 123, 1023, ln }, { 50, 938, ln }, { 40, 1057, ln }, { 68, 263, ln },
    { 87, 370, ln }, { 194, 889, ln }, { 170, 978, ln }, { 165, 1192, ln },
    { 133, 1296, ln },
    //L3 reusable
    { 188, 2.3f, linear }, { 340, 9.2f, linear }, { 158, 604, ln },
    { 159, 813, ln }, { 185, 813, ln }, { 115, 954, ln },
    { 97, 2.29f, linear }, { 186, 5.98f, linear }, { 591, 41, linear },
    { 145, 598, ln }, { 171, 800, ln }, { 144, 901, ln }, { 102, 393, ln },
    { 99, 436, ln }, { 119, 734, ln }, { 136, 985, ln }, { 51, 906, ln },
    { 46, 1008, linear }, { 59, 218, ln }, { 77, 308, ln },
    { 195, 831, ln }, { 179, 911, ln }, { 180, 1112, ln },
    { 139, 1209, ln },
    //L3 opt
    { 79, 309, ln }, { 126, 512, ln }, { 121, 734, ln }, { 114, 827, ln },
    { 4, 798, linear }, { -116, 788, ln }, { 285, 16, linear },
    { 89, 345, ln }, { 173, 788, ln }, { 132, 823, ln }, { 99, 964, ln },
    { -121, 1040, ln }, { 91, 384, ln }, { 120, 552, ln }, { 122, 825, ln },
    { 109, 990, ln }, { -0.18f, 867, linear }, { -55, 910, ln },
    { 65, 253, ln }, { 76, 333, ln }, { 189, 875, ln }, { 158, 927, ln },
    { 137, 1096, ln }, { 84, 1169, ln }
};
// clang-format on

size_t get_ss_util_idx(float v) {
    for (size_t i = 0; i < ss_util_set.size(); i++)
        if (v <= ss_util_set[i]) return i;
    return ss_util_set.size() - 1;
};
size_t get_appr_table_idx(float ss_util, data_type_t dt, mem_operation_t op,
        bool is_reusable, data_location_t location) {
    size_t idx = get_ss_util_idx(ss_util);
    const int ss_dim = (int)ss_util_set.size();
    const int dt_idx = (dt == data_type::f16 || dt == data_type::bf16) ? 1 : 0;
    const int opt_idx = op == mem_operation_t::read ? 0 : 1;
    const int reusable_idx = is_reusable ? 0 : 1;
    const int location_idx = location == HBM ? 0 : 1;
    return idx + dt_idx * ss_dim + opt_idx * (2 * ss_dim)
            + reusable_idx * (2 * 2 * ss_dim)
            + location_idx * (2 * 2 * 2 * ss_dim);
}

float get_appr_val(float a, float b, float x, appr_alg_t alg) {
    if (alg == linear)
        return a * x + b;
    else if (alg == ln)
        return a * log(x) + b;
    else
        gpu_assert(false) << "Unexpected approximation alg";
    return 0.f;
}

float get_thr_utilization_factor(float ss_util, float thr_util,
        data_location_t location, compute::gpu_arch_t gpu_arch,
        mem_operation_t op, data_type_t dt, bool is_reusable) {
    float ss_util_adj = std::min(ss_util, max_appr_ss_util);
    float thr_util_adj = std::min(thr_util, max_appr_thr_util);

    if (is_reusable) {
        const size_t idx = get_appr_table_idx(
                ss_util_adj, dt, op, is_reusable, location);

        const float a = appr_table[idx].a;
        const float b = appr_table[idx].b;
        const appr_alg_t used_alg = appr_table[idx].alg;

        const float y = get_appr_val(a, b, thr_util_adj, used_alg);
        const float y_max = get_appr_val(a, b, max_appr_thr_util, used_alg);
        return y_max / y;
    } else {
        if (location == L3) {
            // for all archs
            float ss_util_adj = std::min(ss_util, 1.0f);
            float thr_util_adj = std::min(thr_util, 1.0f);
            const float y_br = 1 - ss_util_adj / 2;
            return 1.f
                    / solve_2pieces_linear_function(
                            thr_util_adj, 0.f, 0.25f, 1.f, 0.f, y_br, 1.f);
        } else { // HBM
            if (gpu_arch == compute::gpu_arch_t::xe_hpg) {
                const float x_br = pow(
                        2, (log2(utils::rnd_up_pow2((int)round(ss_util))) - 4));
                const float y_br = ss_util > 4 ? 0.9 : 0.5;
                return 1.f
                        / solve_2pieces_linear_function(
                                thr_util, 0.f, x_br, 32, 0.f, y_br, 1.f);

            } else if (gpu_arch >= compute::gpu_arch_t::xe_hpc) {
                float ss_util_adj = std::min(ss_util, 1.0f);
                float thr_util_adj = std::min(thr_util, 1.0f);
                const float y_br = ss_util_adj < 0.25 ? 0.9 : 0.7;
                return 1.f
                        / solve_2pieces_linear_function(
                                thr_util_adj, 0.f, 0.125f, 1.f, 0.f, y_br, 1.f);
            } else {
                assert(!"unsupported");
                return 1.f;
            }
        }
    }
}

bool is_reduction_kernel(const kernel_kind_t &kernel) {
    return kernel == reduce_stats_fwd_ker || kernel == reduce_mean_var_ker
            || kernel == reduce_stats_bwd_ker
            || kernel == reusable_reduce_stats_fwd_ker;
}

void get_estimated_kernel_time(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc) {
    const data_location_t input_location = desc.input_location;
    const data_location_t output_location = desc.output_location;
    const size_t read_nbytes = desc.input_nbytes;
    const size_t write_nbytes = desc.output_nbytes;
    // consider data location.
    float read_ns = read_nbytes
            / (input_location == L3 ? hw_params.L3_bw : hw_params.HBM_bw);
    float write_ns = write_nbytes
            / (output_location == L3 ? hw_params.L3_bw : hw_params.HBM_bw);
    // only for debug print
    float r_ns_base = read_ns;
    float w_ns_base = write_ns;
    MAYBE_UNUSED(r_ns_base);
    MAYBE_UNUSED(w_ns_base);

    // consider HW utilization

    // SS utilization
    const float adj_util = std::min(desc.ss_util, 1.f);
    const float ss_utilization_factor = get_ss_utilization_factor(
            adj_util, conf.data_type, desc.reusable_version);
    read_ns *= ss_utilization_factor;
    write_ns *= ss_utilization_factor;

    // thr utilization
    read_ns *= get_thr_utilization_factor(desc.ss_util, desc.used_ss_thr_util,
            input_location, hw_params.gpu_arch, mem_operation_t::read,
            conf.data_type, desc.reusable_version);
    write_ns *= get_thr_utilization_factor(desc.ss_util, desc.used_ss_thr_util,
            output_location, hw_params.gpu_arch, mem_operation_t::write,
            conf.data_type, desc.reusable_version);

    // consider atomics cost
    if (p.use_fused_atomics_reduction
            && (desc.kernel == calc_mean_ker || desc.kernel == calc_var_ker
                    || desc.kernel == calc_mean_var_ker
                    || desc.kernel == calc_stats_ker)) {
        write_ns *= 64; // based on PVC perf data
    }

    // only for debug print
    float r_ns_location = read_ns;
    float w_ns_location = write_ns;
    MAYBE_UNUSED(r_ns_location);
    MAYBE_UNUSED(w_ns_location);

    // consider vectorization
    const int vect_size
            = is_reduction_kernel(desc.kernel) && desc.reusable_version
            ? def_reduction_vect
            : p.vect_size;

    const float v_coeff = get_vectorization_factor(
            vect_size, conf.data_type, desc.reusable_version);
    read_ns *= v_coeff;
    write_ns *= v_coeff;
    desc.time_ns = read_ns + write_ns;
    // For debuging and analysis purposes
    std::string kernel_type_name = to_string(desc.kernel);
    DPRINT_MODEL(
            "%s:%s:%d estimation - %s : p = %d %d %d : thr_util = %g "
            "ss_util = "
            "%g "
            ": base %.1f %.1f "
            ": location %.1f %.1f "
            ": v_coeff %.1f "
            ": final %.1f %.1f : kernel_total %.1f\n",
            PRINTHEAD, kernel_type_name.c_str(), p.use_fused_atomics_reduction,
            p.ic_block, p.stat_sp_block, desc.used_ss_thr_util, desc.ss_util,
            r_ns_base, w_ns_base, r_ns_location, w_ns_location, v_coeff,
            read_ns, write_ns, desc.time_ns);
}

void init_ker_desc(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc,
        bool reusable_version, const kernel_kind_t kernel) {
    desc.kernel = kernel;
    desc.reusable_version = reusable_version;
    desc.ncalls = get_ncalls(p, conf, kernel);
}
void dump_kernel_desc(kernel_desc_t &desc) {
    std::string kernel_type_name = to_string(desc.kernel);
    DPRINT("%s:%s:%d kernel_desc: %s : reusable = %s : ncalls = %d\n",
            PRINTHEAD, kernel_type_name.c_str(),
            desc.reusable_version ? "yes" : "no", desc.ncalls);
}

void init_kernel_descriptors(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, bool reusable_version) {
    kernel_desc_t desc;

    // logic about which kernels will be running and how many times
    if (conf.is_forward) {
        init_ker_desc(
                p, conf, hw_params, desc, reusable_version, default_fwd_ker);
        p.kernel_descs.push_back(desc);
        if (conf.calculate_stats) {
            if (conf.use_stats_one_pass) {
                init_ker_desc(p, conf, hw_params, desc, reusable_version,
                        calc_mean_var_ker);
                p.kernel_descs.push_back(desc);
            } else {
                init_ker_desc(p, conf, hw_params, desc, reusable_version,
                        calc_mean_ker);
                p.kernel_descs.push_back(desc);
                init_ker_desc(p, conf, hw_params, desc, reusable_version,
                        calc_var_ker);
                p.kernel_descs.push_back(desc);
            }

            if (p.use_fused_atomics_reduction) {
                // distinguished due to different data amount to process
                init_ker_desc(p, conf, hw_params, desc, reusable_version,
                        reduce_aux_init_ker);
                p.kernel_descs.push_back(desc);
                init_ker_desc(p, conf, hw_params, desc, reusable_version,
                        reduce_aux_finalize_ker);
                p.kernel_descs.push_back(desc);
            } else {
                if (conf.use_stats_one_pass) {
                    init_ker_desc(p, conf, hw_params, desc, reusable_version,
                            reduce_mean_var_ker);
                    p.kernel_descs.push_back(desc);
                } else {
                    init_ker_desc(p, conf, hw_params, desc, reusable_version,
                            reusable_version ? reusable_reduce_stats_fwd_ker
                                             : reduce_stats_fwd_ker);
                    p.kernel_descs.push_back(desc);
                }
            }
        }
    } else { // BWD pass
        init_ker_desc(
                p, conf, hw_params, desc, reusable_version, default_bwd_ker);
        p.kernel_descs.push_back(desc);
        init_ker_desc(
                p, conf, hw_params, desc, reusable_version, calc_stats_ker);
        p.kernel_descs.push_back(desc);
        if (p.use_fused_atomics_reduction) {
            init_ker_desc(p, conf, hw_params, desc, reusable_version,
                    reduce_aux_init_ker);
            p.kernel_descs.push_back(desc);
            init_ker_desc(p, conf, hw_params, desc, reusable_version,
                    reduce_aux_finalize_ker);
            p.kernel_descs.push_back(desc);
        } else {
            init_ker_desc(p, conf, hw_params, desc, reusable_version,
                    reduce_stats_bwd_ker);
            p.kernel_descs.push_back(desc);
        }
    }
    dump_kernel_desc(desc);
}

void dump_params(std::vector<model_params_t> &params) {
    DPRINT_MODEL("%s:%s:%d params\n", PRINTHEAD);
    for (auto &p : params) {
        DPRINT_MODEL(
                "use_fused_atomics_reduction = %d ic_block = %d stat_sp_block "
                "= "
                "%d vect_size = %d\n",
                p.use_fused_atomics_reduction, p.ic_block, p.stat_sp_block,
                p.vect_size);
    }
}

status_t get_estimated_hw_utilization(model_params_t &p,
        nhwc_bnorm_params_t &conf, hw_params_t &hw_params,
        kernel_desc_t &desc) {
    auto *compute_engine
            = downcast<compute::compute_engine_t *>(hw_params.engine);
    compute::dispatch_t dry_run_dispatch // to get auto-generated lws
            = compute_engine->create_dispatch();

    nhwc_bnorm_params_t conf_dry_run {conf};
    conf_dry_run.set_use_fused_atomics_reduction(p.use_fused_atomics_reduction);
    conf_dry_run.set_ic_block(p.ic_block);
    conf_dry_run.set_stat_sp_block(p.stat_sp_block);
    conf_dry_run.set_update_sp_block(p.stat_sp_block);
    conf_dry_run.set_update_sp_unroll(1);
    CHECK(nhwc_bnorm_kernel_dispatching(
            desc.kernel, conf_dry_run, hw_params.engine, dry_run_dispatch));

    auto nd_range = dry_run_dispatch.nd_range();
    const compute::range_t gws = nd_range.global_range();
    const compute::range_t lws = nd_range.local_range();
    if (lws.nelems() == 0) return status::runtime_error;
    desc.num_wgs = gws.nelems() / lws.nelems();
    desc.used_ss_thr_util = get_used_ss_thr_utilization(
            hw_params, conf.sub_group_size, gws, lws);
    desc.ss_util = get_ss_utilization(hw_params.max_ss, gws, lws);
    return status::success;
}

status_t make_kernel_perf_estimation(model_params_t &p,
        nhwc_bnorm_params_t &conf, kernel_desc_t &desc,
        hw_params_t &hw_params) {

    CHECK(get_estimated_hw_utilization(p, conf, hw_params, desc));

    desc.input_nbytes = get_kernel_input_size(p, conf, desc);
    desc.output_nbytes = get_kernel_output_size(p, conf, desc);
    get_expected_data_location(p, conf, hw_params, desc);
    dump_kernel_descriptor(desc);

    get_estimated_kernel_time(p, conf, hw_params, desc);
    return status::success;
}

// Make execution time estimation based on data amount, data location and
// HW utilization
status_t make_perf_estimations(
        model_params_t &p, nhwc_bnorm_params_t &conf, hw_params_t &hw_params) {
    for (auto &desc : p.kernel_descs) {
        CHECK(make_kernel_perf_estimation(p, conf, desc, hw_params));
    }
    return status::success;
}

// Get the best set of bnorm parameters based on performance model
// common for nhwc-optimized and nhwc-reusable implementations
status_t get_params_by_model(nhwc_bnorm_params_t &conf,
        const batch_normalization_pd_t *pd, hw_params_t &hw_params,
        bool reusable_version) {

    // Create set of possible parameters
    std::vector<model_params_t> params;
    model_params_t p;
    p.ic_block = conf.sub_group_size;
    assert(conf.ic % conf.sub_group_size == 0);

    while (p.ic_block <= conf.ic
            && (reusable_version ? p.ic_block <= conf.max_ic_block : true)) {
        if (conf.ic % p.ic_block == 0) {
            const dim_t calc_stat_ic = get_nhwc_calc_stat_ic(
                    conf.ic, p.ic_block, conf.sub_group_size);
            p.stat_sp_block = get_nhwc_sp_block_size(conf.sp, calc_stat_ic,
                    hw_params.eu_count, hw_params.threads_per_eu,
                    conf.sub_group_size);
            p.vect_size = get_nhwc_vect_size(p.ic_block, conf.max_vect_size());
            p.use_fused_atomics_reduction = 0;
            params.push_back(p);
            if (hw_params.gpu_arch >= compute::gpu_arch_t::xe_hpc
                    && !pd->attr()->deterministic_) {
                // atomics-based reduction on PVC+ only, perforformance reasons
                p.use_fused_atomics_reduction = 1;
                params.push_back(p);
            }
        }
        p.ic_block += conf.sub_group_size;
    }

    dump_params(params);

    // find the best set
    float best_expected_time = FLT_MAX;
    model_params_t best_params;
    for (auto &p : params) {

        // initialize kernel descriptors
        init_kernel_descriptors(p, conf, hw_params, reusable_version);
        // make estimations on execution time
        CHECK(make_perf_estimations(p, conf, hw_params));

        float exp_time = 0.0f;
        for (auto &desc : p.kernel_descs) {
            exp_time += desc.ncalls * desc.time_ns;
            exp_time += hw_params.host_overheads_per_kernel * desc.ncalls;
            DPRINT_MODEL("%s:%s:%d desc loop: p: %d %d %d : %s: %.1f(%.1f) \n",
                    PRINTHEAD, p.use_fused_atomics_reduction, p.ic_block,
                    p.stat_sp_block, to_string(desc.kernel).c_str(),
                    desc.time_ns, desc.time_ns * desc.ncalls);
        }
        DPRINT_MODEL(
                "%s:%s:%d p: %d %d %d : total expected ns = %.1f ( %.4f ms)\n",
                PRINTHEAD, p.use_fused_atomics_reduction, p.ic_block,
                p.stat_sp_block, exp_time, exp_time * 1e-6);

        if (exp_time < best_expected_time) {
            best_params = p;
            best_expected_time = exp_time;
        }
    }

#define SAVE_PARAM(name, val) \
    if (!conf.name##_param().is_overridden()) conf.set_##name(val);

    // save best params to conf
    conf.expected_time_ms = best_expected_time * 1e-6;
    // Some parameters can be set by tuning procedure or taken from table.
    // Other parametes to be set by model.
    SAVE_PARAM(use_fused_atomics_reduction,
            best_params.use_fused_atomics_reduction);
    if (!conf.ic_block_param().is_overridden()
            // guard for tuning, to use default value if overrrided one is wrong
            || (conf.ic_block_param().is_overridden()
                    && conf.ic_block() > conf.ic))
        conf.set_ic_block(best_params.ic_block);
    conf.calc_stat_ic = get_nhwc_calc_stat_ic(
            conf.ic, conf.ic_block(), conf.sub_group_size);
    SAVE_PARAM(stat_sp_block, best_params.stat_sp_block);
    SAVE_PARAM(update_sp_block, conf.stat_sp_block());
    SAVE_PARAM(update_sp_unroll, 1);

#undef SAVE_PARAM

    conf.vect_size = get_nhwc_vect_size(
            conf.ic_block(), conf.max_vect_size(), conf.sub_group_size);
    // Guard for tuning and lookup table -
    // to use the default value if overrrided one is wrong
    const bool bad_update_sp_unroll
            = conf.update_sp_block() % conf.update_sp_unroll()
            || (conf.sp % conf.update_sp_block()) % conf.update_sp_unroll();
    if (conf.update_sp_unroll_param().is_overridden() && bad_update_sp_unroll) {
        conf.set_update_sp_unroll(1);
    } else {
        assert(!bad_update_sp_unroll);
    }
    return status::success;
}

} // namespace bn_model
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
