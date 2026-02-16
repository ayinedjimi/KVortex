// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/memory/pool.hpp"
#include "kvortex/core/logger.hpp"
#include <cuda_runtime.h>
#include <cstring>

// Optional NUMA support
#ifdef __linux__
#include <numa.h>
#endif

namespace kvortex {

// ============================================================================
// PinnedHostPool Implementation
// ============================================================================

Result<std::unique_ptr<PinnedHostPool>> PinnedHostPool::create(
    size_t total_size,
    bool numa_aware,
    size_t alignment) {

    auto pool = std::unique_ptr<PinnedHostPool>(new PinnedHostPool());
    pool->total_size_ = total_size;
    pool->alignment_ = alignment;
    pool->numa_aware_ = numa_aware;

#ifdef __linux__
    if (numa_aware && numa_available() >= 0) {
        // Allocate on local NUMA node
        numa_set_localalloc();
        LOG_INFO("NUMA-aware allocation enabled (node: local)");
    }
#endif

    // Allocate pinned host memory
    cudaError_t err = cudaHostAlloc(&pool->base_ptr_, total_size,
                                    cudaHostAllocDefault | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to allocate pinned memory: {} bytes, error: {}",
                 total_size, cudaGetErrorString(err));
        return std::unexpected(KVortexError::AllocationFailed);
    }

    // Initialize with single large free block
    pool->free_blocks_.emplace(total_size, 0);

    LOG_INFO("Created pinned host pool: {} GB, alignment: {} bytes",
            total_size / (1024.0 * 1024.0 * 1024.0), alignment);

    return pool;
}

PinnedHostPool::~PinnedHostPool() {
    if (base_ptr_) {
        cudaFreeHost(base_ptr_);
        LOG_DEBUG("Freed pinned host pool: {} bytes", total_size_);
    }
}

PinnedHostPool::PinnedHostPool(PinnedHostPool&& other) noexcept
    : base_ptr_(other.base_ptr_)
    , total_size_(other.total_size_)
    , alignment_(other.alignment_)
    , numa_aware_(other.numa_aware_)
    , free_blocks_(std::move(other.free_blocks_))
    , allocated_blocks_(std::move(other.allocated_blocks_)) {
    other.base_ptr_ = nullptr;
}

PinnedHostPool& PinnedHostPool::operator=(PinnedHostPool&& other) noexcept {
    if (this != &other) {
        if (base_ptr_) {
            cudaFreeHost(base_ptr_);
        }
        base_ptr_ = other.base_ptr_;
        total_size_ = other.total_size_;
        alignment_ = other.alignment_;
        numa_aware_ = other.numa_aware_;
        free_blocks_ = std::move(other.free_blocks_);
        allocated_blocks_ = std::move(other.allocated_blocks_);
        other.base_ptr_ = nullptr;
    }
    return *this;
}

Result<void*> PinnedHostPool::allocate(size_t size) {
    std::lock_guard lock(mutex_);

    // Align size
    size_t aligned_size = (size + alignment_ - 1) & ~(alignment_ - 1);

    // Find first-fit free block
    auto it = free_blocks_.lower_bound(aligned_size);
    if (it == free_blocks_.end()) {
        LOG_WARN("Pinned pool out of memory: requested {} bytes, largest free block: {}",
                aligned_size,
                free_blocks_.empty() ? 0 : free_blocks_.rbegin()->first);
        return std::unexpected(KVortexError::OutOfMemory);
    }

    size_t block_size = it->first;
    size_t block_offset = it->second;

    // Remove from free list
    free_blocks_.erase(it);

    // Calculate pointer
    void* ptr = static_cast<char*>(base_ptr_) + block_offset;

    // Add to allocated blocks
    allocated_blocks_[ptr] = aligned_size;

    // Return remainder to free list
    if (block_size > aligned_size) {
        size_t remainder = block_size - aligned_size;
        size_t remainder_offset = block_offset + aligned_size;
        free_blocks_.emplace(remainder, remainder_offset);
    }

    LOG_TRACE("Allocated {} bytes from pinned pool (ptr: {}, offset: {})",
             aligned_size, ptr, block_offset);

    return ptr;
}

VoidResult PinnedHostPool::deallocate(void* ptr) {
    if (!ptr) {
        return std::unexpected(KVortexError::InvalidPointer);
    }

    std::lock_guard lock(mutex_);

    // Find allocation
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        LOG_ERROR("Attempted to deallocate invalid pointer: {}", ptr);
        return std::unexpected(KVortexError::InvalidPointer);
    }

    size_t size = it->second;
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_ptr_);

    // Remove from allocated list
    allocated_blocks_.erase(it);

    // Return to free list (with coalescing)
    // TODO: Implement coalescing for production version
    free_blocks_.emplace(size, offset);

    LOG_TRACE("Deallocated {} bytes from pinned pool (offset: {})", size, offset);

    return {};
}

PinnedHostPool::Stats PinnedHostPool::get_stats() const {
    std::lock_guard lock(mutex_);

    Stats stats;
    stats.total_size = total_size_;
    stats.num_allocations = allocated_blocks_.size();
    stats.num_free_blocks = free_blocks_.size();

    for (const auto& [ptr, size] : allocated_blocks_) {
        stats.allocated_size += size;
    }

    if (stats.total_size > 0) {
        stats.fragmentation_pct = 100.0f * (1.0f - static_cast<float>(
            free_blocks_.empty() ? 0 : free_blocks_.rbegin()->first) / total_size_);
    }

    return stats;
}

// ============================================================================
// GPUAsyncPool Implementation
// ============================================================================

Result<std::unique_ptr<GPUAsyncPool>> GPUAsyncPool::create(
    size_t total_size,
    cudaStream_t stream,
    int device_id) {

    auto pool = std::unique_ptr<GPUAsyncPool>(new GPUAsyncPool());
    pool->total_size_ = total_size;
    pool->stream_ = stream;
    pool->device_id_ = device_id;

    // Set device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to set CUDA device {}: {}",
                 device_id, cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAError);
    }

    // Create memory pool
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device_id;

    err = cudaMemPoolCreate(&pool->pool_, &props);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to create CUDA memory pool: {}",
                 cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAError);
    }

    // Set release threshold
    uint64_t threshold = total_size;
    err = cudaMemPoolSetAttribute(pool->pool_, cudaMemPoolAttrReleaseThreshold,
                                  &threshold);
    if (err != cudaSuccess) {
        LOG_WARN("Failed to set memory pool threshold: {}",
                cudaGetErrorString(err));
    }

    LOG_INFO("Created GPU async pool: {} GB on device {}",
            total_size / (1024.0 * 1024.0 * 1024.0), device_id);

    return pool;
}

GPUAsyncPool::~GPUAsyncPool() {
    if (pool_) {
        cudaMemPoolDestroy(pool_);
        LOG_DEBUG("Destroyed GPU async pool");
    }
}

GPUAsyncPool::GPUAsyncPool(GPUAsyncPool&& other) noexcept
    : pool_(other.pool_)
    , stream_(other.stream_)
    , device_id_(other.device_id_)
    , total_size_(other.total_size_)
    , allocations_(std::move(other.allocations_)) {
    other.pool_ = nullptr;
}

GPUAsyncPool& GPUAsyncPool::operator=(GPUAsyncPool&& other) noexcept {
    if (this != &other) {
        if (pool_) {
            cudaMemPoolDestroy(pool_);
        }
        pool_ = other.pool_;
        stream_ = other.stream_;
        device_id_ = other.device_id_;
        total_size_ = other.total_size_;
        allocations_ = std::move(other.allocations_);
        other.pool_ = nullptr;
    }
    return *this;
}

Result<void*> GPUAsyncPool::allocate_async(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, stream_);
    if (err != cudaSuccess) {
        LOG_ERROR("GPU async allocation failed: {} bytes, error: {}",
                 size, cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAAllocationFailed);
    }

    {
        std::lock_guard lock(mutex_);
        allocations_[ptr] = size;
    }

    LOG_TRACE("Allocated {} bytes from GPU pool (ptr: {})", size, ptr);

    return ptr;
}

VoidResult GPUAsyncPool::deallocate_async(void* ptr) {
    if (!ptr) {
        return std::unexpected(KVortexError::InvalidPointer);
    }

    cudaError_t err = cudaFreeAsync(ptr, stream_);
    if (err != cudaSuccess) {
        LOG_ERROR("GPU async deallocation failed: {}",
                 cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAError);
    }

    {
        std::lock_guard lock(mutex_);
        allocations_.erase(ptr);
    }

    LOG_TRACE("Deallocated from GPU pool (ptr: {})", ptr);

    return {};
}

GPUAsyncPool::Stats GPUAsyncPool::get_stats() const {
    std::lock_guard lock(mutex_);

    Stats stats;
    stats.total_size = total_size_;
    stats.num_allocations = allocations_.size();

    for (const auto& [ptr, size] : allocations_) {
        stats.allocated_size += size;
    }

    return stats;
}

} // namespace kvortex
