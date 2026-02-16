// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/cache/eviction.hpp"
#include "kvortex/core/logger.hpp"

namespace kvortex {

LRUEvictionPolicy::LRUEvictionPolicy(const Config& config)
    : config_(config) {
    LOG_INFO("LRU eviction policy created: watermark={:.1f}%, eviction_ratio={:.1f}%",
            config_.watermark * 100, config_.eviction_ratio * 100);
}

VoidResult LRUEvictionPolicy::access(const BlockID& id, size_t size) {
    std::lock_guard lock(mutex_);

    auto it = blocks_.find(id);
    if (it != blocks_.end()) {
        // Move to front (most recently used)
        lru_list_.erase(it->second.lru_it);
        lru_list_.push_front(id);
        it->second.lru_it = lru_list_.begin();
    } else {
        // New block
        lru_list_.push_front(id);

        BlockInfo info;
        info.id = id;
        info.size = size;
        info.lru_it = lru_list_.begin();
        blocks_[id] = info;

        current_size_ += size;
    }

    return {};
}

VoidResult LRUEvictionPolicy::mark_evictable(const BlockID& id) {
    std::lock_guard lock(mutex_);

    auto it = blocks_.find(id);
    if (it == blocks_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    it->second.pinned = false;
    return {};
}

VoidResult LRUEvictionPolicy::pin(const BlockID& id) {
    std::lock_guard lock(mutex_);

    auto it = blocks_.find(id);
    if (it == blocks_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    it->second.pinned = true;
    LOG_TRACE("Pinned block: {}", id.to_hex().substr(0, 16));
    return {};
}

VoidResult LRUEvictionPolicy::unpin(const BlockID& id) {
    std::lock_guard lock(mutex_);

    auto it = blocks_.find(id);
    if (it == blocks_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    it->second.pinned = false;
    LOG_TRACE("Unpinned block: {}", id.to_hex().substr(0, 16));
    return {};
}

bool LRUEvictionPolicy::should_evict() const {
    std::lock_guard lock(mutex_);

    if (config_.max_capacity_bytes == 0) {
        return false;
    }

    float utilization = static_cast<float>(current_size_) / config_.max_capacity_bytes;
    return utilization >= config_.watermark;
}

std::vector<BlockID> LRUEvictionPolicy::select_eviction_candidates() {
    std::lock_guard lock(mutex_);

    std::vector<BlockID> candidates;

    // Calculate how much to evict
    size_t target_evict_bytes = static_cast<size_t>(
        config_.max_capacity_bytes * config_.eviction_ratio);

    size_t evicted_bytes = 0;

    // Select from least recently used (back of list)
    for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
        if (evicted_bytes >= target_evict_bytes) {
            break;
        }

        const auto& block_id = *it;
        auto block_it = blocks_.find(block_id);
        if (block_it == blocks_.end()) {
            continue;
        }

        // Skip pinned blocks
        if (block_it->second.pinned) {
            continue;
        }

        candidates.push_back(block_id);
        evicted_bytes += block_it->second.size;
    }

    if (!candidates.empty()) {
        LOG_INFO("Selected {} blocks for eviction ({:.2f} MB)",
                candidates.size(), evicted_bytes / (1024.0 * 1024.0));
    }

    return candidates;
}

VoidResult LRUEvictionPolicy::remove(const BlockID& id) {
    std::lock_guard lock(mutex_);

    auto it = blocks_.find(id);
    if (it == blocks_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    current_size_ -= it->second.size;
    total_evicted_bytes_ += it->second.size;
    num_evictions_++;

    lru_list_.erase(it->second.lru_it);
    blocks_.erase(it);

    return {};
}

void LRUEvictionPolicy::update_size(int64_t delta) {
    std::lock_guard lock(mutex_);
    if (delta < 0 && current_size_ < static_cast<size_t>(-delta)) {
        current_size_ = 0;
    } else {
        current_size_ += delta;
    }
}

LRUEvictionPolicy::Stats LRUEvictionPolicy::get_stats() const {
    std::lock_guard lock(mutex_);

    Stats stats;
    stats.current_size_bytes = current_size_;
    stats.num_blocks = blocks_.size();
    stats.num_evictions = num_evictions_;
    stats.total_evicted_bytes = total_evicted_bytes_;

    return stats;
}

} // namespace kvortex
