// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/cache/index.hpp"
#include "kvortex/core/logger.hpp"
#include <chrono>

namespace kvortex {

// ============================================================================
// BlockHasher Implementation
// ============================================================================

SHA256Hash BlockHasher::hash_tokens(const std::vector<int32_t>& tokens) const {
    return hash_data(tokens.data(), tokens.size() * sizeof(int32_t));
}

SHA256Hash BlockHasher::hash_data(const void* data, size_t size) const {
    SHA256Hash hash;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        LOG_ERROR("Failed to create EVP_MD_CTX");
        return hash;
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        LOG_ERROR("EVP_DigestInit_ex failed");
        EVP_MD_CTX_free(ctx);
        return hash;
    }

    if (EVP_DigestUpdate(ctx, data, size) != 1) {
        LOG_ERROR("EVP_DigestUpdate failed");
        EVP_MD_CTX_free(ctx);
        return hash;
    }

    unsigned int hash_len = 0;
    if (EVP_DigestFinal_ex(ctx, hash.data.data(), &hash_len) != 1) {
        LOG_ERROR("EVP_DigestFinal_ex failed");
    }

    EVP_MD_CTX_free(ctx);
    return hash;
}

std::vector<SHA256Hash> BlockHasher::hash_chunks(
    const std::vector<int32_t>& tokens,
    size_t chunk_size) const {

    std::vector<SHA256Hash> hashes;
    size_t num_chunks = (tokens.size() + chunk_size - 1) / chunk_size;

    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, tokens.size());
        size_t size = end - start;

        auto hash = hash_data(&tokens[start], size * sizeof(int32_t));
        hashes.push_back(hash);
    }

    return hashes;
}

// ============================================================================
// CacheIndex Implementation
// ============================================================================

VoidResult CacheIndex::insert(const BlockID& id, const BlockLocation& location) {
    std::unique_lock lock(mutex_);

    IndexEntry entry;
    entry.location = location;
    entry.last_access_time = get_timestamp_us();
    entry.access_count = 0;

    index_[id] = entry;

    LOG_TRACE("Inserted block into index: {} (tier: {}, offset: {})",
             id.to_hex().substr(0, 16), static_cast<int>(location.tier), location.offset);

    return {};
}

Result<BlockLocation> CacheIndex::lookup(const BlockID& id) {
    num_lookups_++;

    std::shared_lock lock(mutex_);

    auto it = index_.find(id);
    if (it == index_.end()) {
        num_misses_++;
        return std::unexpected(KVortexError::BlockNotFound);
    }

    num_hits_++;

    // Update access time (requires upgrade to unique lock)
    lock.unlock();
    std::unique_lock ulock(mutex_);

    it = index_.find(id);  // Re-find after lock upgrade
    if (it != index_.end()) {
        it->second.last_access_time = get_timestamp_us();
        it->second.access_count++;
        return it->second.location;
    }

    return std::unexpected(KVortexError::BlockNotFound);
}

bool CacheIndex::contains(const BlockID& id) const {
    std::shared_lock lock(mutex_);
    return index_.find(id) != index_.end();
}

std::vector<bool> CacheIndex::check_blocks(const std::vector<BlockID>& ids) const {
    std::shared_lock lock(mutex_);

    std::vector<bool> results;
    results.reserve(ids.size());

    for (const auto& id : ids) {
        results.push_back(index_.find(id) != index_.end());
    }

    return results;
}

VoidResult CacheIndex::remove(const BlockID& id) {
    std::unique_lock lock(mutex_);

    auto it = index_.find(id);
    if (it == index_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    index_.erase(it);

    LOG_TRACE("Removed block from index: {}", id.to_hex().substr(0, 16));

    return {};
}

VoidResult CacheIndex::touch(const BlockID& id) {
    std::unique_lock lock(mutex_);

    auto it = index_.find(id);
    if (it == index_.end()) {
        return std::unexpected(KVortexError::BlockNotFound);
    }

    it->second.last_access_time = get_timestamp_us();
    it->second.access_count++;

    return {};
}

std::vector<BlockID> CacheIndex::get_all_blocks() const {
    std::shared_lock lock(mutex_);

    std::vector<BlockID> blocks;
    blocks.reserve(index_.size());

    for (const auto& [id, _] : index_) {
        blocks.push_back(id);
    }

    return blocks;
}

CacheIndex::Stats CacheIndex::get_stats() const {
    Stats stats;

    {
        std::shared_lock lock(mutex_);
        stats.num_blocks = index_.size();
    }

    stats.num_lookups = num_lookups_.load();
    stats.num_hits = num_hits_.load();
    stats.num_misses = num_misses_.load();

    if (stats.num_lookups > 0) {
        stats.hit_rate = static_cast<double>(stats.num_hits) / stats.num_lookups;
    }

    return stats;
}

void CacheIndex::clear() {
    std::unique_lock lock(mutex_);
    index_.clear();
    LOG_INFO("Cache index cleared");
}

int64_t CacheIndex::get_timestamp_us() {
    auto now = ::std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return ::std::chrono::duration_cast<::std::chrono::microseconds>(duration).count();
}

} // namespace kvortex
