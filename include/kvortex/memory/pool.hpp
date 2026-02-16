// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../core/error.hpp"
#include "../core/types.hpp"
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace kvortex {

// ============================================================================
// Pinned Host Memory Pool (NUMA-aware)
// ============================================================================

class PinnedHostPool {
public:
    struct Stats {
        size_t total_size{0};
        size_t allocated_size{0};
        size_t num_allocations{0};
        size_t num_free_blocks{0};
        float fragmentation_pct{0.0f};
    };

    /// Create pool with specified total size
    /// @param total_size Total pool size in bytes
    /// @param numa_aware Enable NUMA-aware allocation (requires libnuma)
    /// @param alignment Memory alignment (default 128 bytes)
    static Result<std::unique_ptr<PinnedHostPool>> create(
        size_t total_size,
        bool numa_aware = true,
        size_t alignment = 128);

    ~PinnedHostPool();

    // Delete copy, allow move
    PinnedHostPool(const PinnedHostPool&) = delete;
    PinnedHostPool& operator=(const PinnedHostPool&) = delete;
    PinnedHostPool(PinnedHostPool&&) noexcept;
    PinnedHostPool& operator=(PinnedHostPool&&) noexcept;

    /// Allocate memory from pool
    /// @param size Size in bytes
    /// @return Pointer to allocated memory or error
    Result<void*> allocate(size_t size);

    /// Deallocate memory back to pool
    /// @param ptr Pointer to free (must have been allocated from this pool)
    VoidResult deallocate(void* ptr);

    /// Get pool statistics
    Stats get_stats() const;

    /// Get base pointer (for debugging)
    void* base_ptr() const noexcept { return base_ptr_; }

    /// Get total pool size
    size_t total_size() const noexcept { return total_size_; }

private:
    PinnedHostPool() = default;

    void* base_ptr_{nullptr};
    size_t total_size_{0};
    size_t alignment_{128};
    bool numa_aware_{false};

    // Free blocks: size -> offset
    std::multimap<size_t, size_t> free_blocks_;

    // Allocated blocks: ptr -> size
    std::unordered_map<void*, size_t> allocated_blocks_;

    mutable std::mutex mutex_;
};

// ============================================================================
// GPU Async Memory Pool (Stream-Ordered Allocation)
// ============================================================================

class GPUAsyncPool {
public:
    struct Stats {
        size_t total_size{0};
        size_t allocated_size{0};
        size_t num_allocations{0};
    };

    /// Create GPU async pool
    /// @param total_size Total pool size in bytes
    /// @param stream CUDA stream for stream-ordered allocations
    /// @param device_id CUDA device ID
    static Result<std::unique_ptr<GPUAsyncPool>> create(
        size_t total_size,
        cudaStream_t stream,
        int device_id = 0);

    ~GPUAsyncPool();

    // Delete copy, allow move
    GPUAsyncPool(const GPUAsyncPool&) = delete;
    GPUAsyncPool& operator=(const GPUAsyncPool&) = delete;
    GPUAsyncPool(GPUAsyncPool&&) noexcept;
    GPUAsyncPool& operator=(GPUAsyncPool&&) noexcept;

    /// Allocate GPU memory asynchronously
    /// @param size Size in bytes
    /// @return Pointer to GPU memory or error
    Result<void*> allocate_async(size_t size);

    /// Deallocate GPU memory asynchronously
    /// @param ptr Pointer to free
    VoidResult deallocate_async(void* ptr);

    /// Get pool statistics
    Stats get_stats() const;

    /// Get associated CUDA stream
    cudaStream_t stream() const noexcept { return stream_; }

    /// Get total pool size
    size_t total_size() const noexcept { return total_size_; }

private:
    GPUAsyncPool() = default;

    cudaMemPool_t pool_{nullptr};
    cudaStream_t stream_{nullptr};
    int device_id_{0};
    size_t total_size_{0};

    // Track allocations for stats
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;
};

} // namespace kvortex
