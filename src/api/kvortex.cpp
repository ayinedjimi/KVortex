// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/api/kvortex.hpp"
#include "kvortex/core/logger.hpp"

namespace kvortex {

Result<std::unique_ptr<KVortexEngine>> KVortexEngine::create(const KVortexConfig& config) {
    auto engine = std::unique_ptr<KVortexEngine>(new KVortexEngine());
    engine->config_ = config;

    // Configure logging
    Logger::instance().set_level(config.log_level);
    if (!config.log_file.empty()) {
        Logger::instance().set_output_file(config.log_file);
    }

    LOG_INFO("Creating KVortex engine...");

    // Create CPU pool
    auto cpu_pool_result = PinnedHostPool::create(
        config.cpu_pool_size_bytes,
        config.enable_numa,
        config.alignment_bytes);
    if (!cpu_pool_result) {
        return std::unexpected(cpu_pool_result.error());
    }
    engine->cpu_pool_ = std::shared_ptr<PinnedHostPool>(std::move(*cpu_pool_result));

    // Create stream manager
    auto stream_mgr_result = StreamManager::create(
        config.num_transfer_streams,
        config.cuda_device_id);
    if (!stream_mgr_result) {
        return std::unexpected(stream_mgr_result.error());
    }
    engine->stream_manager_ = std::shared_ptr<StreamManager>(std::move(*stream_mgr_result));

    // Create cache index
    engine->cache_index_ = std::make_shared<CacheIndex>();

    // Create eviction policy
    LRUEvictionPolicy::Config eviction_config;
    eviction_config.watermark = config.eviction_watermark;
    eviction_config.eviction_ratio = config.eviction_ratio;
    eviction_config.max_capacity_bytes = config.cpu_pool_size_bytes;
    engine->eviction_policy_ = std::make_shared<LRUEvictionPolicy>(eviction_config);

    // Create CPU backend
    auto cpu_backend_result = CPUBackend::create(engine->cpu_pool_);
    if (!cpu_backend_result) {
        return std::unexpected(cpu_backend_result.error());
    }
    engine->cpu_backend_ = std::shared_ptr<CPUBackend>(std::move(*cpu_backend_result));

    LOG_INFO("KVortex engine created successfully");
    LOG_INFO("  CPU pool: {} GB", config.cpu_pool_size_bytes / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  GPU pool: {} GB", config.gpu_pool_size_bytes / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Transfer streams: {}", config.num_transfer_streams);
    LOG_INFO("  Thread pool: {} workers", config.thread_pool_size);

    return engine;
}

KVortexEngine::~KVortexEngine() {
    if (!shutdown_) {
        shutdown();
    }
}

KVortexEngine::KVortexEngine(KVortexEngine&& other) noexcept
    : config_(other.config_)
    , cpu_pool_(std::move(other.cpu_pool_))
    , stream_manager_(std::move(other.stream_manager_))
    , cache_index_(std::move(other.cache_index_))
    , eviction_policy_(std::move(other.eviction_policy_))
    , cpu_backend_(std::move(other.cpu_backend_))
    , shutdown_(other.shutdown_) {
    other.shutdown_ = true;
}

KVortexEngine& KVortexEngine::operator=(KVortexEngine&& other) noexcept {
    if (this != &other) {
        shutdown();
        config_ = other.config_;
        cpu_pool_ = std::move(other.cpu_pool_);
        stream_manager_ = std::move(other.stream_manager_);
        cache_index_ = std::move(other.cache_index_);
        eviction_policy_ = std::move(other.eviction_policy_);
        cpu_backend_ = std::move(other.cpu_backend_);
        shutdown_ = other.shutdown_;
        other.shutdown_ = true;
    }
    return *this;
}

VoidResult KVortexEngine::save_blocks(
    const std::vector<BlockID>& block_ids,
    const std::vector<const void*>& data,
    const std::vector<size_t>& sizes) {

    if (block_ids.size() != data.size() || block_ids.size() != sizes.size()) {
        return std::unexpected(KVortexError::InvalidArgument);
    }

    for (size_t i = 0; i < block_ids.size(); ++i) {
        // Save to CPU backend
        auto result = cpu_backend_->save(block_ids[i], data[i], sizes[i]);
        if (!result) {
            LOG_ERROR("Failed to save block {}", block_ids[i].to_hex().substr(0, 16));
            return result;
        }

        // Add to cache index
        BlockLocation location{StorageTier::CPU, 0, sizes[i]};
        cache_index_->insert(block_ids[i], location);

        // Track in eviction policy
        eviction_policy_->access(block_ids[i], sizes[i]);
    }

    // Check if eviction is needed
    if (eviction_policy_->should_evict()) {
        auto candidates = eviction_policy_->select_eviction_candidates();
        for (const auto& id : candidates) {
            cpu_backend_->remove(id);
            cache_index_->remove(id);
            eviction_policy_->remove(id);
        }
    }

    return {};
}

VoidResult KVortexEngine::load_blocks(
    const std::vector<BlockID>& block_ids,
    const std::vector<void*>& data,
    const std::vector<size_t>& sizes) {

    if (block_ids.size() != data.size() || block_ids.size() != sizes.size()) {
        return std::unexpected(KVortexError::InvalidArgument);
    }

    for (size_t i = 0; i < block_ids.size(); ++i) {
        // Load from CPU backend
        auto result = cpu_backend_->load(block_ids[i], data[i], sizes[i]);
        if (!result) {
            LOG_ERROR("Failed to load block {}", block_ids[i].to_hex().substr(0, 16));
            return result;
        }

        // Update access time
        eviction_policy_->access(block_ids[i], sizes[i]);
    }

    return {};
}

std::vector<bool> KVortexEngine::check_blocks(const std::vector<BlockID>& block_ids) const {
    return cache_index_->check_blocks(block_ids);
}

Result<AsyncHandle> KVortexEngine::save_blocks_async(
    const std::vector<BlockID>& block_ids,
    const std::vector<const void*>& data,
    const std::vector<size_t>& sizes) {

    // For now, just do synchronous save
    // TODO: Implement true async with thread pool
    auto result = save_blocks(block_ids, data, sizes);
    if (!result) {
        return std::unexpected(result.error());
    }

    return 0;  // Dummy handle
}

VoidResult KVortexEngine::wait(AsyncHandle /*handle*/) {
    // For now, immediate return (synchronous implementation)
    return {};
}

CacheStats KVortexEngine::get_stats() const {
    CacheStats stats;

    auto index_stats = cache_index_->get_stats();
    stats.num_cached_blocks = index_stats.num_blocks;
    stats.num_hits = index_stats.num_hits;
    stats.num_misses = index_stats.num_misses;
    stats.cache_hit_rate = index_stats.hit_rate;

    auto backend_stats = cpu_backend_->get_stats();
    stats.cpu_bytes = backend_stats.total_bytes;

    auto eviction_stats = eviction_policy_->get_stats();
    stats.num_evictions = eviction_stats.num_evictions;

    return stats;
}

void KVortexEngine::shutdown() {
    if (shutdown_) {
        return;
    }

    LOG_INFO("Shutting down KVortex engine...");

    // Synchronize all streams
    if (stream_manager_) {
        stream_manager_->synchronize_all();
    }

    shutdown_ = true;

    LOG_INFO("KVortex engine shutdown complete");
}

} // namespace kvortex
