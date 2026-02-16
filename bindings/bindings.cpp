// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kvortex/core/types.hpp"
#include "kvortex/core/config.hpp"

namespace py = pybind11;
using namespace kvortex;

// ============================================================================
// Python Module (placeholder for full API)
// ============================================================================

PYBIND11_MODULE(kvortex_cpp, m) {
    m.doc() = "KVortex: High-Performance KV Cache for vLLM";

    // Core types
    py::class_<KVortexConfig>(m, "Config")
        .def(py::init<>())
        .def_readwrite("gpu_pool_size_bytes", &KVortexConfig::gpu_pool_size_bytes)
        .def_readwrite("cpu_pool_size_bytes", &KVortexConfig::cpu_pool_size_bytes)
        .def_readwrite("num_transfer_streams", &KVortexConfig::num_transfer_streams)
        .def_readwrite("thread_pool_size", &KVortexConfig::thread_pool_size)
        .def_readwrite("chunk_size", &KVortexConfig::chunk_size)
        .def_readwrite("eviction_watermark", &KVortexConfig::eviction_watermark)
        .def_readwrite("enable_numa", &KVortexConfig::enable_numa)
        .def_readwrite("disk_cache_dir", &KVortexConfig::disk_cache_dir);

    // Statistics
    py::class_<CacheStats>(m, "Stats")
        .def(py::init<>())
        .def_readonly("num_cached_blocks", &CacheStats::num_cached_blocks)
        .def_readonly("total_bytes_cached", &CacheStats::total_bytes_cached)
        .def_readonly("cache_hit_rate", &CacheStats::cache_hit_rate)
        .def_readonly("avg_load_latency_ms", &CacheStats::avg_load_latency_ms)
        .def_readonly("avg_save_latency_ms", &CacheStats::avg_save_latency_ms);

    // KVortexEngine class (simplified for now - full integration in Phase 4)
    // py::class_<KVortexEngine>(m, "KVortexEngine")
    //     .def_static("create", &KVortexEngine::create)
    //     .def("save_blocks", &KVortexEngine::save_blocks, py::call_guard<py::gil_scoped_release>())
    //     .def("load_blocks", &KVortexEngine::load_blocks, py::call_guard<py::gil_scoped_release>())
    //     .def("check_blocks", &KVortexEngine::check_blocks)
    //     .def("get_stats", &KVortexEngine::get_stats)
    //     .def("shutdown", &KVortexEngine::shutdown);

    m.attr("__version__") = "1.0.0";
}
