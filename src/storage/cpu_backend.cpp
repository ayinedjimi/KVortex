// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/storage/cpu_backend.hpp"
#include "kvortex/core/logger.hpp"
#include <cstring>

namespace kvortex {

Result<std::unique_ptr<CPUBackend>> CPUBackend::create(
    std::shared_ptr<PinnedHostPool> pool) {

    if (!pool) {
        return std::unexpected(KVortexError::InvalidArgument);
    }

    auto backend = std::unique_ptr<CPUBackend>(new CPUBackend());
    backend->pool_ = pool;

    LOG_INFO("Created CPU backend with pinned memory pool");
    return backend;
}

VoidResult CPUBackend::save(const BlockID& id, const void* data, size_t size) {
    // Allocate from pool
    auto ptr_result = pool_->allocate(size);
    if (!ptr_result) {
        LOG_ERROR("Failed to allocate {} bytes from CPU pool", size);
        return std::unexpected(ptr_result.error());
    }

    void* ptr = *ptr_result;

    // Copy data
    std::memcpy(ptr, data, size);

    // Store in map
    {
        std::unique_lock lock(mutex_);
        blocks_[id] = BlockData{ptr, size};
    }

    num_saves_++;

    LOG_TRACE("Saved block to CPU: {} ({} bytes)", id.to_hex().substr(0, 16), size);

    return {};
}

VoidResult CPUBackend::load(const BlockID& id, void* data, size_t size) {
    BlockData block_data;

    {
        std::shared_lock lock(mutex_);
        auto it = blocks_.find(id);
        if (it == blocks_.end()) {
            return std::unexpected(KVortexError::BlockNotFound);
        }
        block_data = it->second;
    }

    if (block_data.size != size) {
        LOG_ERROR("Size mismatch: expected {} bytes, got {}", size, block_data.size);
        return std::unexpected(KVortexError::InvalidArgument);
    }

    // Copy data
    std::memcpy(data, block_data.ptr, size);

    num_loads_++;

    LOG_TRACE("Loaded block from CPU: {} ({} bytes)", id.to_hex().substr(0, 16), size);

    return {};
}

bool CPUBackend::contains(const BlockID& id) const {
    std::shared_lock lock(mutex_);
    return blocks_.find(id) != blocks_.end();
}

VoidResult CPUBackend::remove(const BlockID& id) {
    std::unique_lock lock(mutex_);

    auto it = blocks_.find(id);
    if (it == blocks_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    // Deallocate from pool
    auto result = pool_->deallocate(it->second.ptr);
    if (!result) {
        LOG_WARN("Failed to deallocate block from pool");
    }

    blocks_.erase(it);

    LOG_TRACE("Removed block from CPU: {}", id.to_hex().substr(0, 16));

    return {};
}

Result<size_t> CPUBackend::get_size(const BlockID& id) const {
    std::shared_lock lock(mutex_);

    auto it = blocks_.find(id);
    if (it == blocks_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    return it->second.size;
}

StorageBackend::Stats CPUBackend::get_stats() const {
    Stats stats;

    {
        std::shared_lock lock(mutex_);
        stats.num_blocks = blocks_.size();
        for (const auto& [id, data] : blocks_) {
            stats.total_bytes += data.size;
        }
    }

    stats.num_saves = num_saves_.load();
    stats.num_loads = num_loads_.load();

    return stats;
}

} // namespace kvortex
