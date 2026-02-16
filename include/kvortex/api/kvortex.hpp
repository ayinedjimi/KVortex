// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../core/config.hpp"
#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../memory/pool.hpp"
#include "../transfer/stream_manager.hpp"
#include "../cache/index.hpp"
#include "../cache/eviction.hpp"
#include "../storage/cpu_backend.hpp"
#include <memory>

namespace kvortex {

// ============================================================================
// KVortex Engine - Main API
// ============================================================================

class KVortexEngine {
public:
    /// Create KVortex engine
    /// @param config Configuration
    /// @return Engine instance or error
    static Result<std::unique_ptr<KVortexEngine>> create(const KVortexConfig& config);

    ~KVortexEngine();

    // Delete copy, allow move
    KVortexEngine(const KVortexEngine&) = delete;
    KVortexEngine& operator=(const KVortexEngine&) = delete;
    KVortexEngine(KVortexEngine&&) noexcept;
    KVortexEngine& operator=(KVortexEngine&&) noexcept;

    /// Save blocks to cache
    /// @param block_ids Vector of block IDs
    /// @param data Vector of data pointers (must match block_ids size)
    /// @param sizes Vector of sizes (must match block_ids size)
    /// @return Success or error
    VoidResult save_blocks(
        const std::vector<BlockID>& block_ids,
        const std::vector<const void*>& data,
        const std::vector<size_t>& sizes);

    /// Load blocks from cache
    /// @param block_ids Vector of block IDs
    /// @param data Vector of output buffers (must be pre-allocated)
    /// @param sizes Expected sizes
    /// @return Success or error
    VoidResult load_blocks(
        const std::vector<BlockID>& block_ids,
        const std::vector<void*>& data,
        const std::vector<size_t>& sizes);

    /// Check which blocks are cached (bitmask query)
    /// @param block_ids Vector of block IDs to check
    /// @return Vector of booleans (true = cached)
    std::vector<bool> check_blocks(const std::vector<BlockID>& block_ids) const;

    /// Save blocks asynchronously
    /// @return AsyncHandle for waiting
    Result<AsyncHandle> save_blocks_async(
        const std::vector<BlockID>& block_ids,
        const std::vector<const void*>& data,
        const std::vector<size_t>& sizes);

    /// Wait for async operation
    /// @param handle AsyncHandle from async operation
    VoidResult wait(AsyncHandle handle);

    /// Get statistics
    CacheStats get_stats() const;

    /// Shutdown engine
    void shutdown();

private:
    KVortexEngine() = default;

    // Components
    KVortexConfig config_;
    std::shared_ptr<PinnedHostPool> cpu_pool_;
    std::shared_ptr<StreamManager> stream_manager_;
    std::shared_ptr<CacheIndex> cache_index_;
    std::shared_ptr<LRUEvictionPolicy> eviction_policy_;
    std::shared_ptr<CPUBackend> cpu_backend_;

    bool shutdown_{false};
};

} // namespace kvortex
