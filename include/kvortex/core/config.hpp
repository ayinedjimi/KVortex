// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace kvortex {

// ============================================================================
// Configuration
// ============================================================================

struct KVortexConfig {
    // Memory configuration
    size_t gpu_pool_size_bytes{8ULL * 1024 * 1024 * 1024};  ///< 8GB default
    size_t cpu_pool_size_bytes{16ULL * 1024 * 1024 * 1024}; ///< 16GB default
    size_t disk_pool_size_bytes{100ULL * 1024 * 1024 * 1024}; ///< 100GB default
    bool enable_numa{true};  ///< NUMA-aware allocation

    // Transfer configuration
    int num_transfer_streams{3};  ///< Number of CUDA streams for transfers
    size_t transfer_batch_size{32};  ///< Max requests per batch
    size_t transfer_batch_bytes{128ULL * 1024 * 1024};  ///< 128MB batch threshold
    bool enable_double_buffering{true};

    // Threading configuration
    int thread_pool_size{8};  ///< Worker threads
    bool enable_scheduler_thread{true};
    int scheduler_poll_interval_us{100};  ///< Scheduler polling interval

    // Cache configuration
    size_t chunk_size{256};  ///< Tokens per chunk (for hashing)
    float eviction_watermark{0.8f};  ///< Trigger eviction at 80% capacity
    float eviction_ratio{0.2f};  ///< Evict 20% when triggered
    bool enable_prefetch{false};

    // Storage backends
    bool enable_cpu_backend{true};
    bool enable_disk_backend{true};
    bool enable_redis_backend{false};
    bool enable_s3_backend{false};
    std::string disk_cache_dir{"/tmp/kvortex_cache"};
    std::string redis_url{"redis://localhost:6379"};
    std::string s3_bucket;

    // vLLM integration
    uint32_t default_block_size{16};  ///< Tokens per block (vLLM default)

    // Logging and monitoring
    enum class LogLevel { TRACE, DEBUG, INFO, WARN, ERROR };
    LogLevel log_level{LogLevel::INFO};
    bool enable_metrics{true};
    std::string log_file;  ///< Empty = stderr

    // Advanced options
    bool enable_compression{false};  ///< CacheGen compression
    int cuda_device_id{0};
    size_t alignment_bytes{128};  ///< Memory alignment (cache line size)
};

} // namespace kvortex
