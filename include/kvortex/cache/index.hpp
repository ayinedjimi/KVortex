// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../core/error.hpp"
#include "../core/types.hpp"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace kvortex {

// ============================================================================
// SHA256 Block Hasher
// ============================================================================

class BlockHasher {
public:
    BlockHasher() = default;

    /// Compute SHA256 hash of token sequence
    /// @param tokens Vector of token IDs
    /// @return SHA256 hash (32 bytes)
    SHA256Hash hash_tokens(const std::vector<int32_t>& tokens) const;

    /// Compute SHA256 hash of raw data
    /// @param data Pointer to data
    /// @param size Size in bytes
    /// @return SHA256 hash
    SHA256Hash hash_data(const void* data, size_t size) const;

    /// Compute chunked hash (for LMCache compatibility)
    /// @param tokens All tokens
    /// @param chunk_size Tokens per chunk (default: 256)
    /// @return Vector of chunk hashes
    std::vector<SHA256Hash> hash_chunks(
        const std::vector<int32_t>& tokens,
        size_t chunk_size = 256) const;
};

// ============================================================================
// Cache Index (Thread-Safe)
// ============================================================================

class CacheIndex {
public:
    struct Stats {
        size_t num_blocks{0};
        size_t num_lookups{0};
        size_t num_hits{0};
        size_t num_misses{0};
        double hit_rate{0.0};
    };

    CacheIndex() = default;

    /// Insert block into index
    /// @param id Block ID (SHA256 hash)
    /// @param location Block storage location
    VoidResult insert(const BlockID& id, const BlockLocation& location);

    /// Lookup block location
    /// @param id Block ID
    /// @return Block location or error if not found
    Result<BlockLocation> lookup(const BlockID& id);

    /// Check if block exists (without incrementing hit counter)
    /// @param id Block ID
    /// @return true if block exists
    bool contains(const BlockID& id) const;

    /// Check multiple blocks (bitmask query for vLLM)
    /// @param ids Vector of block IDs
    /// @return Vector of booleans (true = cached, false = not cached)
    std::vector<bool> check_blocks(const std::vector<BlockID>& ids) const;

    /// Remove block from index
    /// @param id Block ID
    VoidResult remove(const BlockID& id);

    /// Update block metadata (for LRU tracking)
    /// @param id Block ID
    VoidResult touch(const BlockID& id);

    /// Get all block IDs
    std::vector<BlockID> get_all_blocks() const;

    /// Get statistics
    Stats get_stats() const;

    /// Clear all entries
    void clear();

private:
    struct IndexEntry {
        BlockLocation location;
        int64_t last_access_time{0};
        uint64_t access_count{0};
    };

    mutable std::shared_mutex mutex_;
    std::unordered_map<BlockID, IndexEntry> index_;

    // Statistics
    mutable std::atomic<size_t> num_lookups_{0};
    mutable std::atomic<size_t> num_hits_{0};
    mutable std::atomic<size_t> num_misses_{0};

    static int64_t get_timestamp_us();
};

} // namespace kvortex
