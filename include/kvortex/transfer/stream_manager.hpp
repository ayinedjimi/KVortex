// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../core/error.hpp"
#include "../core/types.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace kvortex {

// ============================================================================
// Multi-Stream Transfer Engine
// ============================================================================

class StreamManager {
public:
    struct Stats {
        uint64_t total_transfers{0};
        uint64_t total_bytes_transferred{0};
        double avg_bandwidth_gbps{0.0};
        double avg_latency_ms{0.0};
    };

    /// Create stream manager with specified number of streams
    /// @param num_streams Number of CUDA streams (3+ recommended)
    /// @param device_id CUDA device ID
    static Result<std::unique_ptr<StreamManager>> create(
        int num_streams = 3,
        int device_id = 0);

    ~StreamManager();

    // Delete copy, allow move
    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;
    StreamManager(StreamManager&&) noexcept;
    StreamManager& operator=(StreamManager&&) noexcept;

    /// Copy GPU → CPU asynchronously
    /// @param dst CPU destination (must be pinned memory)
    /// @param src GPU source
    /// @param size Size in bytes
    /// @param stream_idx Stream index (0 to num_streams-1)
    /// @return AsyncHandle for synchronization
    Result<AsyncHandle> copy_gpu_to_cpu_async(
        void* dst, const void* src, size_t size, int stream_idx = 0);

    /// Copy CPU → GPU asynchronously
    /// @param dst GPU destination
    /// @param src CPU source (must be pinned memory)
    /// @param size Size in bytes
    /// @param stream_idx Stream index
    /// @return AsyncHandle for synchronization
    Result<AsyncHandle> copy_cpu_to_gpu_async(
        void* dst, const void* src, size_t size, int stream_idx = 0);

    /// Check if transfer is complete
    /// @param handle AsyncHandle returned from copy operation
    bool is_transfer_complete(AsyncHandle handle) const;

    /// Wait for transfer to complete
    /// @param handle AsyncHandle to wait for
    VoidResult wait_for_transfer(AsyncHandle handle);

    /// Synchronize specific stream
    /// @param stream_idx Stream index to synchronize
    VoidResult synchronize_stream(int stream_idx);

    /// Synchronize all streams
    VoidResult synchronize_all();

    /// Get CUDA stream handle
    cudaStream_t get_stream(int stream_idx) const;

    /// Get number of streams
    int num_streams() const noexcept { return num_streams_; }

    /// Get statistics
    Stats get_stats() const;

private:
    StreamManager() = default;

    int device_id_{0};
    int num_streams_{0};
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;

    // Statistics
    mutable std::atomic<uint64_t> total_transfers_{0};
    mutable std::atomic<uint64_t> total_bytes_{0};

    // Handle generation
    std::atomic<AsyncHandle> next_handle_{0};
    mutable std::mutex handle_mutex_;
    std::unordered_map<AsyncHandle, int> handle_to_stream_;
};

// ============================================================================
// Transfer Batch Queue
// ============================================================================

struct TransferRequest {
    void* dst;
    const void* src;
    size_t size;
    bool is_gpu_to_cpu;  // true = GPU→CPU, false = CPU→GPU
    AsyncHandle handle;
};

class BatchQueue {
public:
    /// Create batch queue
    /// @param stream_manager Stream manager for executing transfers
    /// @param max_batch_size Maximum requests per batch (default: 32)
    /// @param max_batch_bytes Maximum bytes per batch (default: 128MB)
    static Result<std::unique_ptr<BatchQueue>> create(
        std::shared_ptr<StreamManager> stream_manager,
        size_t max_batch_size = 32,
        size_t max_batch_bytes = 128ULL * 1024 * 1024);

    ~BatchQueue() = default;

    /// Enqueue transfer request
    /// @param req Transfer request
    VoidResult enqueue(TransferRequest req);

    /// Flush pending requests immediately
    VoidResult flush();

    /// Get number of pending requests
    size_t pending_count() const;

private:
    BatchQueue() = default;

    std::shared_ptr<StreamManager> stream_manager_;
    size_t max_batch_size_;
    size_t max_batch_bytes_;

    mutable std::mutex mutex_;
    std::vector<TransferRequest> pending_;
    size_t total_pending_bytes_{0};

    VoidResult flush_locked();
};

} // namespace kvortex
