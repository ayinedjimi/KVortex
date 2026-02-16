// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace kvortex {

// ============================================================================
// Block Identification
// ============================================================================

/// SHA256 hash type (32 bytes)
struct SHA256Hash {
    std::array<uint8_t, 32> data;

    bool operator==(const SHA256Hash& other) const noexcept {
        return data == other.data;
    }

    bool operator!=(const SHA256Hash& other) const noexcept {
        return data != other.data;
    }

    /// Convert to hex string for debugging
    std::string to_hex() const;
};

/// Block identifier (content-addressable via SHA256)
using BlockID = SHA256Hash;

// ============================================================================
// Storage Tiers
// ============================================================================

enum class StorageTier : uint8_t {
    GPU,      ///< GPU VRAM (fastest, most limited)
    CPU,      ///< Pinned host memory (fast PCIe transfers)
    Disk,     ///< Local NVMe/SSD storage
    Remote    ///< Remote storage (Redis, S3, etc.)
};

/// Block location in storage hierarchy
struct BlockLocation {
    StorageTier tier;
    uint64_t offset;        ///< Offset within tier (bytes)
    size_t size;            ///< Block size (bytes)

    bool operator==(const BlockLocation& other) const noexcept {
        return tier == other.tier && offset == other.offset && size == other.size;
    }
};

// ============================================================================
// Block Metadata
// ============================================================================

struct BlockMetadata {
    BlockID id;
    BlockLocation location;
    uint64_t access_count{0};
    int64_t last_access_time{0};  ///< Timestamp (microseconds since epoch)
    uint32_t ref_count{0};        ///< Reference count for eviction
    bool pinned{false};           ///< If true, cannot be evicted

    /// Model-specific metadata
    uint32_t num_layers{0};
    uint32_t num_kv_heads{0};
    uint32_t head_dim{0};
    uint16_t block_size{16};      ///< Tokens per block (vLLM default: 16)

    /// Data type (0=FP32, 1=FP16, 2=BF16, 3=FP8)
    uint8_t dtype{1};  // Default to FP16
};

// ============================================================================
// Tensor View (Zero-Copy Tensor Wrapper)
// ============================================================================

/// Lightweight tensor view (non-owning pointer to data)
struct TensorView {
    void* data{nullptr};
    std::vector<int64_t> shape;
    size_t element_size{0};  ///< Bytes per element
    bool is_contiguous{true};

    /// Total size in bytes
    size_t size_bytes() const noexcept {
        size_t total = element_size;
        for (auto dim : shape) {
            total *= static_cast<size_t>(dim);
        }
        return total;
    }

    /// Number of elements
    size_t numel() const noexcept {
        size_t total = 1;
        for (auto dim : shape) {
            total *= static_cast<size_t>(dim);
        }
        return total;
    }
};

// ============================================================================
// vLLM-Specific Types
// ============================================================================

/// vLLM KV block format: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
/// Dimension 0: 2 = [K, V] tensors
/// Dimension 1: Number of transformer layers
/// Dimension 2: Number of KV blocks allocated
/// Dimension 3: Block size (tokens per block, typically 16)
/// Dimension 4: Number of KV attention heads
/// Dimension 5: Head dimension (embedding size per head)
struct VLLMBlockFormat {
    uint32_t num_layers;
    uint32_t num_blocks;
    uint32_t block_size;     ///< Typically 16 tokens
    uint32_t num_kv_heads;
    uint32_t head_dim;

    /// Calculate expected shape for vLLM tensors
    std::vector<int64_t> get_shape() const {
        return {2,
                static_cast<int64_t>(num_layers),
                static_cast<int64_t>(num_blocks),
                static_cast<int64_t>(block_size),
                static_cast<int64_t>(num_kv_heads),
                static_cast<int64_t>(head_dim)};
    }

    /// Calculate bytes per block (all layers, K+V)
    size_t bytes_per_block(size_t element_size) const noexcept {
        return 2 * num_layers * block_size * num_kv_heads * head_dim * element_size;
    }
};

// ============================================================================
// Request Types
// ============================================================================

enum class RequestType : uint8_t {
    Save,       ///< Save blocks to cache
    Load,       ///< Load blocks from cache
    Check,      ///< Check if blocks are cached (bitmask query)
    Evict       ///< Explicit eviction request
};

/// Async operation handle
using AsyncHandle = uint64_t;

// ============================================================================
// Statistics
// ============================================================================

struct CacheStats {
    uint64_t num_cached_blocks{0};
    uint64_t total_bytes_cached{0};
    uint64_t num_hits{0};
    uint64_t num_misses{0};
    double cache_hit_rate{0.0};

    // Tier-specific stats
    uint64_t gpu_bytes{0};
    uint64_t cpu_bytes{0};
    uint64_t disk_bytes{0};
    uint64_t remote_bytes{0};

    // Performance metrics
    double avg_load_latency_ms{0.0};
    double avg_save_latency_ms{0.0};
    double gpu_to_cpu_bandwidth_gbps{0.0};

    // Resource metrics
    uint64_t num_evictions{0};
    double pool_fragmentation_pct{0.0};
};

} // namespace kvortex

// ============================================================================
// Hash Support for std::unordered_map
// ============================================================================

namespace std {
    template<>
    struct hash<kvortex::SHA256Hash> {
        size_t operator()(const kvortex::SHA256Hash& h) const noexcept {
            // Use first 8 bytes as hash (SHA256 is already well-distributed)
            size_t result;
            std::memcpy(&result, h.data.data(), sizeof(size_t));
            return result;
        }
    };
}
