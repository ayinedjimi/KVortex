// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../core/error.hpp"
#include "../core/types.hpp"
#include <memory>
#include <string>

namespace kvortex {

// ============================================================================
// Abstract Storage Backend
// ============================================================================

class StorageBackend {
public:
    struct Stats {
        size_t num_blocks{0};
        size_t total_bytes{0};
        uint64_t num_saves{0};
        uint64_t num_loads{0};
        double avg_save_latency_ms{0.0};
        double avg_load_latency_ms{0.0};
    };

    virtual ~StorageBackend() = default;

    /// Save block to storage
    /// @param id Block ID
    /// @param data Block data
    /// @param size Data size in bytes
    /// @return Success or error
    virtual VoidResult save(const BlockID& id, const void* data, size_t size) = 0;

    /// Load block from storage
    /// @param id Block ID
    /// @param data Buffer to load into (must be pre-allocated)
    /// @param size Expected size
    /// @return Success or error
    virtual VoidResult load(const BlockID& id, void* data, size_t size) = 0;

    /// Check if block exists
    /// @param id Block ID
    /// @return true if block exists
    virtual bool contains(const BlockID& id) const = 0;

    /// Remove block from storage
    /// @param id Block ID
    /// @return Success or error
    virtual VoidResult remove(const BlockID& id) = 0;

    /// Get block size
    /// @param id Block ID
    /// @return Size in bytes or error
    virtual Result<size_t> get_size(const BlockID& id) const = 0;

    /// Get statistics
    virtual Stats get_stats() const = 0;

    /// Get storage tier
    virtual StorageTier tier() const = 0;

    /// Get backend name
    virtual std::string name() const = 0;
};

} // namespace kvortex
