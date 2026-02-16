// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.hpp"
#include "../memory/pool.hpp"
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

namespace kvortex {

// ============================================================================
// CPU Backend (Pinned Host Memory)
// ============================================================================

class CPUBackend : public StorageBackend {
public:
    /// Create CPU backend
    /// @param pool Pinned host memory pool
    static Result<std::unique_ptr<CPUBackend>> create(
        std::shared_ptr<PinnedHostPool> pool);

    ~CPUBackend() override = default;

    // StorageBackend interface
    VoidResult save(const BlockID& id, const void* data, size_t size) override;
    VoidResult load(const BlockID& id, void* data, size_t size) override;
    bool contains(const BlockID& id) const override;
    VoidResult remove(const BlockID& id) override;
    Result<size_t> get_size(const BlockID& id) const override;
    Stats get_stats() const override;
    StorageTier tier() const override { return StorageTier::CPU; }
    std::string name() const override { return "CPU"; }

private:
    CPUBackend() = default;

    struct BlockData {
        void* ptr;
        size_t size;
    };

    std::shared_ptr<PinnedHostPool> pool_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<BlockID, BlockData> blocks_;

    // Statistics
    std::atomic<uint64_t> num_saves_{0};
    std::atomic<uint64_t> num_loads_{0};
};

} // namespace kvortex
