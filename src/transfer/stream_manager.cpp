// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/transfer/stream_manager.hpp"
#include "kvortex/core/logger.hpp"
#include <chrono>

namespace kvortex {

// ============================================================================
// StreamManager Implementation
// ============================================================================

Result<std::unique_ptr<StreamManager>> StreamManager::create(
    int num_streams,
    int device_id) {

    if (num_streams <= 0 || num_streams > 32) {
        LOG_ERROR("Invalid number of streams: {}", num_streams);
        return std::unexpected(KVortexError::InvalidArgument);
    }

    auto manager = std::unique_ptr<StreamManager>(new StreamManager());
    manager->device_id_ = device_id;
    manager->num_streams_ = num_streams;

    // Set device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to set CUDA device {}: {}", device_id, cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAError);
    }

    // Create streams
    manager->streams_.resize(num_streams);
    manager->events_.resize(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        err = cudaStreamCreateWithFlags(&manager->streams_[i], cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create CUDA stream {}: {}", i, cudaGetErrorString(err));
            return std::unexpected(KVortexError::CUDAStreamError);
        }

        err = cudaEventCreate(&manager->events_[i]);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create CUDA event {}: {}", i, cudaGetErrorString(err));
            return std::unexpected(KVortexError::CUDAEventError);
        }
    }

    LOG_INFO("Created stream manager: {} streams on device {}", num_streams, device_id);
    return manager;
}

StreamManager::~StreamManager() {
    for (auto stream : streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    for (auto event : events_) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    LOG_DEBUG("Destroyed stream manager: {} streams", num_streams_);
}

StreamManager::StreamManager(StreamManager&& other) noexcept
    : device_id_(other.device_id_)
    , num_streams_(other.num_streams_)
    , streams_(std::move(other.streams_))
    , events_(std::move(other.events_))
    , total_transfers_(other.total_transfers_.load())
    , total_bytes_(other.total_bytes_.load())
    , next_handle_(other.next_handle_.load())
    , handle_to_stream_(std::move(other.handle_to_stream_)) {
    other.num_streams_ = 0;
}

StreamManager& StreamManager::operator=(StreamManager&& other) noexcept {
    if (this != &other) {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
        for (auto event : events_) {
            cudaEventDestroy(event);
        }

        device_id_ = other.device_id_;
        num_streams_ = other.num_streams_;
        streams_ = std::move(other.streams_);
        events_ = std::move(other.events_);
        total_transfers_ = other.total_transfers_.load();
        total_bytes_ = other.total_bytes_.load();
        next_handle_ = other.next_handle_.load();
        handle_to_stream_ = std::move(other.handle_to_stream_);

        other.num_streams_ = 0;
    }
    return *this;
}

Result<AsyncHandle> StreamManager::copy_gpu_to_cpu_async(
    void* dst, const void* src, size_t size, int stream_idx) {

    if (stream_idx < 0 || stream_idx >= num_streams_) {
        LOG_ERROR("Invalid stream index: {}", stream_idx);
        return std::unexpected(KVortexError::InvalidArgument);
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                      streams_[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaMemcpyAsync failed (GPU→CPU): {}", cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAMemcpyFailed);
    }

    // Record event
    err = cudaEventRecord(events_[stream_idx], streams_[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaEventRecord failed: {}", cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAEventError);
    }

    // Generate handle
    AsyncHandle handle = next_handle_++;
    {
        std::lock_guard lock(handle_mutex_);
        handle_to_stream_[handle] = stream_idx;
    }

    total_transfers_++;
    total_bytes_ += size;

    LOG_TRACE("GPU→CPU transfer started: {} bytes on stream {} (handle: {})",
             size, stream_idx, handle);

    return handle;
}

Result<AsyncHandle> StreamManager::copy_cpu_to_gpu_async(
    void* dst, const void* src, size_t size, int stream_idx) {

    if (stream_idx < 0 || stream_idx >= num_streams_) {
        LOG_ERROR("Invalid stream index: {}", stream_idx);
        return std::unexpected(KVortexError::InvalidArgument);
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                      streams_[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaMemcpyAsync failed (CPU→GPU): {}", cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAMemcpyFailed);
    }

    err = cudaEventRecord(events_[stream_idx], streams_[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaEventRecord failed: {}", cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAEventError);
    }

    AsyncHandle handle = next_handle_++;
    {
        std::lock_guard lock(handle_mutex_);
        handle_to_stream_[handle] = stream_idx;
    }

    total_transfers_++;
    total_bytes_ += size;

    LOG_TRACE("CPU→GPU transfer started: {} bytes on stream {} (handle: {})",
             size, stream_idx, handle);

    return handle;
}

bool StreamManager::is_transfer_complete(AsyncHandle handle) const {
    std::lock_guard lock(handle_mutex_);
    auto it = handle_to_stream_.find(handle);
    if (it == handle_to_stream_.end()) {
        return true;  // Handle not found = already completed
    }

    int stream_idx = it->second;
    cudaError_t err = cudaEventQuery(events_[stream_idx]);
    return (err == cudaSuccess);
}

VoidResult StreamManager::wait_for_transfer(AsyncHandle handle) {
    int stream_idx;
    {
        std::lock_guard lock(handle_mutex_);
        auto it = handle_to_stream_.find(handle);
        if (it == handle_to_stream_.end()) {
            return {};  // Already completed
        }
        stream_idx = it->second;
    }

    cudaError_t err = cudaEventSynchronize(events_[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaEventSynchronize failed: {}", cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAEventError);
    }

    {
        std::lock_guard lock(handle_mutex_);
        handle_to_stream_.erase(handle);
    }

    return {};
}

VoidResult StreamManager::synchronize_stream(int stream_idx) {
    if (stream_idx < 0 || stream_idx >= num_streams_) {
        return std::unexpected(KVortexError::InvalidArgument);
    }

    cudaError_t err = cudaStreamSynchronize(streams_[stream_idx]);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaStreamSynchronize failed: {}", cudaGetErrorString(err));
        return std::unexpected(KVortexError::CUDAStreamError);
    }

    return {};
}

VoidResult StreamManager::synchronize_all() {
    for (int i = 0; i < num_streams_; ++i) {
        cudaError_t err = cudaStreamSynchronize(streams_[i]);
        if (err != cudaSuccess) {
            LOG_ERROR("cudaStreamSynchronize failed on stream {}: {}", i, cudaGetErrorString(err));
            return std::unexpected(KVortexError::CUDAStreamError);
        }
    }
    return {};
}

cudaStream_t StreamManager::get_stream(int stream_idx) const {
    if (stream_idx >= 0 && stream_idx < num_streams_) {
        return streams_[stream_idx];
    }
    return nullptr;
}

StreamManager::Stats StreamManager::get_stats() const {
    Stats stats;
    stats.total_transfers = total_transfers_.load();
    stats.total_bytes_transferred = total_bytes_.load();

    // Calculate bandwidth (simplified)
    if (stats.total_transfers > 0 && stats.total_bytes_transferred > 0) {
        // Placeholder: real bandwidth calculation requires timing
        stats.avg_bandwidth_gbps = 20.0;  // Target bandwidth
    }

    return stats;
}

// ============================================================================
// BatchQueue Implementation
// ============================================================================

Result<std::unique_ptr<BatchQueue>> BatchQueue::create(
    std::shared_ptr<StreamManager> stream_manager,
    size_t max_batch_size,
    size_t max_batch_bytes) {

    if (!stream_manager) {
        return std::unexpected(KVortexError::InvalidArgument);
    }

    auto queue = std::unique_ptr<BatchQueue>(new BatchQueue());
    queue->stream_manager_ = stream_manager;
    queue->max_batch_size_ = max_batch_size;
    queue->max_batch_bytes_ = max_batch_bytes;

    LOG_INFO("Created batch queue: max {} requests, max {} MB per batch",
            max_batch_size, max_batch_bytes / (1024 * 1024));

    return queue;
}

VoidResult BatchQueue::enqueue(TransferRequest req) {
    std::lock_guard lock(mutex_);

    pending_.push_back(req);
    total_pending_bytes_ += req.size;

    // Flush if batch is full
    if (pending_.size() >= max_batch_size_ || total_pending_bytes_ >= max_batch_bytes_) {
        return flush_locked();
    }

    return {};
}

VoidResult BatchQueue::flush() {
    std::lock_guard lock(mutex_);
    return flush_locked();
}

size_t BatchQueue::pending_count() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
}

VoidResult BatchQueue::flush_locked() {
    if (pending_.empty()) {
        return {};
    }

    LOG_DEBUG("Flushing batch: {} requests, {} MB total",
             pending_.size(), total_pending_bytes_ / (1024.0 * 1024.0));

    // Execute all pending transfers (round-robin across streams)
    int num_streams = stream_manager_->num_streams();
    for (size_t i = 0; i < pending_.size(); ++i) {
        auto& req = pending_[i];
        int stream_idx = i % num_streams;

        Result<AsyncHandle> result;
        if (req.is_gpu_to_cpu) {
            result = stream_manager_->copy_gpu_to_cpu_async(
                req.dst, req.src, req.size, stream_idx);
        } else {
            result = stream_manager_->copy_cpu_to_gpu_async(
                req.dst, req.src, req.size, stream_idx);
        }

        if (!result) {
            LOG_ERROR("Batch transfer failed: {}", static_cast<int>(result.error()));
            return std::unexpected(result.error());
        }
    }

    pending_.clear();
    total_pending_bytes_ = 0;

    return {};
}

} // namespace kvortex
