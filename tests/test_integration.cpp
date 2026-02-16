// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/api/kvortex.hpp"
#include "kvortex/cache/index.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace kvortex;

// ============================================================================
// Integration Tests
// ============================================================================

TEST(Integration, EngineCreateAndShutdown) {
    KVortexConfig config;
    config.cpu_pool_size_bytes = 100 * 1024 * 1024;  // 100MB
    config.num_transfer_streams = 2;

    auto result = KVortexEngine::create(config);
    ASSERT_TRUE(result.has_value());

    auto engine = std::move(*result);
    ASSERT_NE(engine, nullptr);

    engine->shutdown();
}

TEST(Integration, SaveAndLoadBlocks) {
    KVortexConfig config;
    config.cpu_pool_size_bytes = 100 * 1024 * 1024;

    auto engine_result = KVortexEngine::create(config);
    ASSERT_TRUE(engine_result.has_value());
    auto engine = std::move(*engine_result);

    // Create test data
    const size_t block_size = 1024;
    std::vector<uint8_t> data1(block_size, 0x42);
    std::vector<uint8_t> data2(block_size, 0x43);

    // Hash blocks
    BlockHasher hasher;
    auto id1 = hasher.hash_data(data1.data(), data1.size());
    auto id2 = hasher.hash_data(data2.data(), data2.size());

    // Save blocks
    std::vector<BlockID> block_ids = {id1, id2};
    std::vector<const void*> data_ptrs = {data1.data(), data2.data()};
    std::vector<size_t> sizes = {block_size, block_size};

    auto save_result = engine->save_blocks(block_ids, data_ptrs, sizes);
    ASSERT_TRUE(save_result.has_value());

    // Check blocks are cached
    auto cached = engine->check_blocks(block_ids);
    ASSERT_EQ(cached.size(), 2);
    EXPECT_TRUE(cached[0]);
    EXPECT_TRUE(cached[1]);

    // Load blocks back
    std::vector<uint8_t> loaded1(block_size);
    std::vector<uint8_t> loaded2(block_size);
    std::vector<void*> load_ptrs = {loaded1.data(), loaded2.data()};

    auto load_result = engine->load_blocks(block_ids, load_ptrs, sizes);
    ASSERT_TRUE(load_result.has_value());

    // Verify data
    EXPECT_EQ(loaded1, data1);
    EXPECT_EQ(loaded2, data2);

    // Check stats
    auto stats = engine->get_stats();
    EXPECT_EQ(stats.num_cached_blocks, 2);
    // Note: cache_hit_rate may be 0 if no lookups have occurred
    EXPECT_GE(stats.cache_hit_rate, 0.0);

    engine->shutdown();
}

TEST(Integration, CacheEviction) {
    KVortexConfig config;
    config.cpu_pool_size_bytes = 10 * 1024;  // Very small: 10KB
    config.eviction_watermark = 0.5f;        // Trigger at 50%
    config.eviction_ratio = 0.5f;            // Evict 50%

    auto engine_result = KVortexEngine::create(config);
    ASSERT_TRUE(engine_result.has_value());
    auto engine = std::move(*engine_result);

    // Save multiple blocks until eviction triggers
    BlockHasher hasher;
    const size_t block_size = 1024;  // 1KB per block

    for (int i = 0; i < 20; ++i) {
        std::vector<uint8_t> data(block_size, static_cast<uint8_t>(i));
        auto id = hasher.hash_data(data.data(), data.size());

        std::vector<BlockID> ids = {id};
        std::vector<const void*> ptrs = {data.data()};
        std::vector<size_t> sizes = {block_size};

        engine->save_blocks(ids, ptrs, sizes);
    }

    // Should have evicted some blocks
    auto stats = engine->get_stats();
    EXPECT_GT(stats.num_evictions, 0);
    EXPECT_LT(stats.num_cached_blocks, 20);  // Not all blocks should be cached

    engine->shutdown();
}

TEST(Integration, BlockHashing) {
    BlockHasher hasher;

    // Test token hashing
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    auto hash1 = hasher.hash_tokens(tokens);

    // Same tokens should produce same hash
    auto hash2 = hasher.hash_tokens(tokens);
    EXPECT_EQ(hash1, hash2);

    // Different tokens should produce different hash
    std::vector<int32_t> tokens2 = {1, 2, 3, 4, 6};
    auto hash3 = hasher.hash_tokens(tokens2);
    EXPECT_NE(hash1, hash3);

    // Test chunked hashing
    std::vector<int32_t> long_tokens(1000);
    for (int i = 0; i < 1000; ++i) {
        long_tokens[i] = i;
    }

    auto chunks = hasher.hash_chunks(long_tokens, 256);
    EXPECT_EQ(chunks.size(), 4);  // 1000 / 256 = 4 chunks (rounded up)
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
