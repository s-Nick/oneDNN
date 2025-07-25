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

#include "gpu/intel/jit/v2/conv/planner/planner.hpp"

#include "oneapi/dnnl/dnnl_config.h"

#include "common/primitive_cache.hpp"
#include "gpu/intel/jit/v2/conv/model.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/conv/plan_registry.hpp"
#include "gpu/intel/jit/v2/conv/planner/bench.hpp"
#include "gpu/intel/jit/v2/conv/planner/model_fit.hpp"
#include "gpu/intel/jit/v2/conv/planner/search.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

static planner_params_t params;

bool find_remove(const char *arg, std::string &s) {
    auto pos = s.find(arg);
    if (pos == std::string::npos) return false;
    s.replace(pos, std::strlen(arg), "");
    return true;
}

std::string find_remove_key_value_impl(const std::string &key, std::string &s) {
    auto i = s.find(key);
    if (i == std::string::npos) return {};
    auto j = key.find(" ", i + key.length());
    if (j == std::string::npos) j = s.length();
    i += key.length();
    auto value = s.substr(i, j - i);
    s.replace(i, j - i, "");
    return value;
}

std::string find_remove_key_value(const std::string &key, std::string &s) {
    auto value = find_remove_key_value_impl("--" + key + " ", s);
    if (!value.empty()) return value;
    return find_remove_key_value_impl(key + "=", s);
}

void print_help() {
    std::cout
            << R"(Usage: gpu_conv_planner [--help] [--bench] [--search] [--auto-search] [kernel descriptor arguments]

Optional arguments:
  --help                Shows help message and exits.
  --bench               Runs benchmarking with provided kernel descriptor.
  --search              Runs search, iterate through missing kernel descriptor properties.
  --auto-search         Runs auto-search to rebuild kernel registry.

)";
    std::cout << "Kernel descriptor arguments:" << std::endl;
    kernel_desc_t::show_help();
}

void init_params(
        int argc, const char **argv, const bench_manager_t &bench_mger) {
    ostringstream_t oss;
    for (int i = 1; i < argc; i++)
        oss << " " << argv[i];
    auto cmd_args = oss.str();
    bool has_bench = find_remove("--bench", cmd_args);
    bool has_search = find_remove("--search", cmd_args);
    bool has_auto_search = find_remove("--auto-search", cmd_args);
    bool has_help = (argc == 1) || find_remove("--help", cmd_args);
    auto s_model = find_remove_key_value("model", cmd_args);

    if (has_help) {
        print_help();
        exit(0);
    }

    int mode_count = 0;
    mode_count += (int)has_bench;
    mode_count += (int)has_search;
    mode_count += (int)has_auto_search;
    if (mode_count > 1) {
        std::cout << "Error: --bench, --search and --auto-search are exclusive."
                  << std::endl;
        exit(1);
    }
    if (has_bench) {
        params.mode = planner_mode_t::bench;
    } else if (has_search) {
        params.mode = planner_mode_t::search;
    } else if (has_auto_search) {
        params.mode = planner_mode_t::auto_search;
    } else {
        params.mode = planner_mode_t::trace;
    }
    switch (params.mode) {
        case planner_mode_t::auto_search: return;
        case planner_mode_t::search:
            if (cmd_args.find("--iter") == std::string::npos) {
                cmd_args += " --iter x";
            }
            break;
        default: break;
    }
    auto &iface = kernel_desc_t::parse_iface();
    iface.parse(cmd_args, params.desc, &params.parse_result);
    params.desc.set_missing();
    if (!s_model.empty()) params.model_set = jit::parse<model_set_t>(s_model);
}

void DNNL_API planner_main(int argc, const char **argv) {
    auto status = set_primitive_cache_capacity(0, 1024);
    if (status != status::success) {
        std::cout << "Error: cannot set primitive cache capacity\n";
        exit(1);
    }
    bench_manager_t bench_mger;
    init_params(argc, argv, bench_mger);
    switch (params.mode) {
        case planner_mode_t::trace: {
            plan_t plan = create_conv_plan(params.desc, bench_mger.hw());
            if (!plan) {
                std::cout << "Error: cannot create plan\n";
                exit(1);
            }
            std::cout << plan.str() << std::endl;
            std::cout << ir_utils::add_tag("Reqs", params.desc.reqs().str())
                      << std::endl;
            if (!params.model_set.is_empty()) {
                std::cout << ir_utils::add_tag("Model", params.model_set.str())
                          << std::endl;
            }
            break;
        }
        case planner_mode_t::bench: {
            auto entry = prepare_plan_registry_entry(bench_mger, params.desc);
            std::cout << entry.str() << std::endl;
            std::cout << "Kernel registry entry:\n  " << entry.registry_str()
                      << std::endl;
            break;
        }
        case planner_mode_t::auto_search:
        case planner_mode_t::search: {
            plan_registry() = plan_registry_t();
            search(bench_mger, params);
            break;
        }
        default: gpu_error_not_expected();
    }
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
