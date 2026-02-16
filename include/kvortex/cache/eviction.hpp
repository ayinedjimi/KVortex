// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../core/error.hpp"
#include "../core/types.hpp"
#include <list>
#include <mutex>
#include <unordered_map>

namespace kvortex {

// ============================================================================
// LRU Eviction Policy
// ============================================================================

class LRUEvictionPolicy {
public:
    struct Config {
        float watermark{0.8f};      ///< Trigger eviction at 80% capacity
        float eviction_ratio{0.2f}; ///< Evict 20% when triggered
        size_t max_capacity_bytes{0};
    };

    struct Stats {
        size_t current_size_bytes{0};
        size_t num_blocks{0};
        size_t num_evictions{0};
        size_t total_evicted_bytes{0};
    };

    explicit LRUEvictionPolicy(const Config& config);

    /// Track block access (moves to front of LRU list)
    /// @param id Block ID
    /// @param size Block size in bytes
    VoidResult access(const BlockID& id, size_t size);

    /// Mark block as evictable
    /// @param id Block ID
    VoidResult mark_evictable(const BlockID& id);

    /// Mark block as pinned (cannot be evicted)
    /// @param id Block ID
    VoidResult pin(const BlockID& id);

    /// Unpin block
    /// @param id Block ID
    VoidResult unpin(const BlockID& id);

    /// Check if eviction should be triggered
    /// @return true if current size exceeds watermark
    bool should_evict() const;

    /// Select blocks to evict
    /// @return Vector of block IDs to evict (LRU order)
    std::vector<BlockID> select_eviction_candidates();

    /// Remove block from tracking
    /// @param id Block ID
    VoidResult remove(const BlockID& id);

    /// Update current size
    /// @param delta Change in size (can be negative)
    void update_size(int64_t delta);

    /// Get statistics
    Stats get_stats() const;

private:
    struct BlockInfo {
        BlockID id;
        size_t size;
        bool pinned{false};
        std::list<BlockID>::iterator lru_it;
    };

    Config config_;
    mutable std::mutex mutex_;

    // LRU list (front = most recently used, back = least recently used)
    std::list<BlockID> lru_list_;

    // Block metadata
    std::unordered_map<BlockID, BlockInfo> blocks_;

    // Statistics
    size_t current_size_{0};
    size_t num_evictions_{0};
    size_t total_evicted_bytes_{0};
};

} // namespace kvortex
