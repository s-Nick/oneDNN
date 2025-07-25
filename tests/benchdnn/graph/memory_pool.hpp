/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef MEMORY_POOL_HPP
#define MEMORY_POOL_HPP

#include <mutex>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define OCL_CHECK(x) \
    do { \
        cl_int s = (x); \
        if (s != CL_SUCCESS) { \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "] '" << #x \
                      << "' failed (status code: " << s << ")." << std::endl; \
            exit(1); \
        } \
    } while (0)

inline void *ocl_malloc_device(
        size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
    using F = void *(*)(cl_context, cl_device_id, cl_ulong *, size_t, cl_uint,
            cl_int *);
    if (size == 0) return nullptr;

    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clDeviceMemAllocINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    cl_int err;
    void *p = f(ctx, dev, nullptr, size, static_cast<cl_uint>(alignment), &err);
    OCL_CHECK(err);
    return p;
}

inline void ocl_free(
        void *ptr, cl_device_id dev, cl_context ctx, cl_event event) {
    if (nullptr == ptr) return;
    using F = cl_int (*)(cl_context, void *);
    if (event) { OCL_CHECK(clWaitForEvents(1, &event)); }
    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clMemBlockingFreeINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    OCL_CHECK(f(ctx, ptr));
}
#endif

inline void *host_malloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif /* _WIN32 */
    return (rc == 0) ? ptr : nullptr;
}

inline void host_free(void *ptr) {
#ifdef _WIN32
    _aligned_free((void *)ptr);
#else
    ::free((void *)ptr);
#endif /* _WIN32 */
}
// This memory pool is for benchdnn graph performance validation. `clear` and
// `set_capacity` functions aren't thread safe.
// Note: memory pool for GPU backend currently.

class simple_memory_pool_t {
public:
    bool check_allocated_mem(void *&ptr, size_t size) {
        // find alloc mm with same size
        const auto cnt = map_size_ptr_.count(size);
        if (cnt > 0) {
            const auto Iter = map_size_ptr_.equal_range(size);
            for (auto it = Iter.first; it != Iter.second; ++it) {
                // check if same size mm is free
                if (is_free_ptr_[it->second.get()]) {
                    ptr = it->second.get();
                    is_free_ptr_[ptr] = false;
                    return false;
                }
            }
        }
        return true;
    }

    // for host memory
    void *allocate(size_t size, size_t alignment) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // fake malloc for 0 size
        if (size == 0) return nullptr;
        void *ptr {nullptr};
        bool need_alloc_new_mm = check_allocated_mem(ptr, size);
        if (need_alloc_new_mm) {
            auto sh_ptr = std::shared_ptr<void> {
                    host_malloc(size, alignment), host_free};
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(size, sh_ptr);
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    void *allocate(
            size_t size, size_t alignment, const void *dev, const void *ctx) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // fake malloc for 0 size
        if (size == 0) return nullptr;

        void *ptr {nullptr};
        bool need_alloc_new_mm = check_allocated_mem(ptr, size);
        if (need_alloc_new_mm) {
            auto sh_ptr = std::shared_ptr<void> {
                    malloc_device(size, *static_cast<const sycl::device *>(dev),
                            *static_cast<const sycl::context *>(ctx)),
                    sycl_deletor_t {*static_cast<const sycl::context *>(ctx)}};
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(size, sh_ptr);
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void *allocate(
            size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // fake malloc for 0 size
        if (size == 0) return nullptr;

        void *ptr {nullptr};
        bool need_alloc_new_mm = check_allocated_mem(ptr, size);
        if (need_alloc_new_mm) {
            auto sh_ptr = std::shared_ptr<void> {
                    ocl_malloc_device(size, alignment, dev, ctx),
                    ocl_deletor_t {dev, ctx}};
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(size, sh_ptr);
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }
#endif

    void deallocate(void *ptr) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        is_free_ptr_[ptr] = true;
    }

    void clear() {
        map_size_ptr_.clear();
        is_free_ptr_.clear();
    }

private:
    std::mutex pool_lock;
    std::unordered_multimap<size_t, std::shared_ptr<void>> map_size_ptr_;
    std::unordered_map<void *, bool> is_free_ptr_;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    struct sycl_deletor_t {
        sycl_deletor_t() = delete;
        sycl_deletor_t(const ::sycl::context &ctx) : ctx_(ctx) {}
        void operator()(void *ptr) {
            if (ptr) ::sycl::free(ptr, ctx_);
        }

    private:
        ::sycl::context ctx_;
    };
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    struct ocl_deletor_t {
        ocl_deletor_t() = delete;
        ocl_deletor_t(cl_device_id dev, cl_context ctx)
            : dev_(dev), ctx_(ctx) {}
        void operator()(void *ptr) {
            if (ptr) ocl_free(ptr, dev_, ctx_, {});
        }

    private:
        cl_device_id dev_;
        cl_context ctx_;
    };
#endif
};

#endif
