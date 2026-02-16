// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// SPDX-License-Identifier: Apache-2.0

#include "kvortex/memory/pool.hpp"
#include <gtest/gtest.h>

using namespace kvortex;

// ============================================================================
// Pinned Host Pool Tests
// ============================================================================

TEST(PinnedHostPool, CreatePool) {
    constexpr size_t pool_size = 1024 * 1024;  // 1 MB
    auto result = PinnedHostPool::create(pool_size, false);
    ASSERT_TRUE(result.has_value());

    auto pool = std::move(*result);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->total_size(), pool_size);
}

TEST(PinnedHostPool, AllocateAndDeallocate) {
    auto pool_result = PinnedHostPool::create(10 * 1024 * 1024, false);  // 10 MB
    ASSERT_TRUE(pool_result.has_value());
    auto pool = std::move(*pool_result);

    // Allocate 1 MB
    auto ptr1_result = pool->allocate(1024 * 1024);
    ASSERT_TRUE(ptr1_result.has_value());
    void* ptr1 = *ptr1_result;
    ASSERT_NE(ptr1, nullptr);

    // Allocate another 2 MB
    auto ptr2_result = pool->allocate(2 * 1024 * 1024);
    ASSERT_TRUE(ptr2_result.has_value());
    void* ptr2 = *ptr2_result;
    ASSERT_NE(ptr2, nullptr);

    // Check stats
    auto stats = pool->get_stats();
    EXPECT_EQ(stats.num_allocations, 2);
    EXPECT_GT(stats.allocated_size, 0);

    // Deallocate
    EXPECT_TRUE(pool->deallocate(ptr1).has_value());
    EXPECT_TRUE(pool->deallocate(ptr2).has_value());

    // Should be able to allocate again
    auto ptr3_result = pool->allocate(3 * 1024 * 1024);
    EXPECT_TRUE(ptr3_result.has_value());
}

TEST(PinnedHostPool, OutOfMemory) {
    auto pool_result = PinnedHostPool::create(1024 * 1024, false);  // 1 MB
    ASSERT_TRUE(pool_result.has_value());
    auto pool = std::move(*pool_result);

    // Try to allocate more than available
    auto result = pool->allocate(10 * 1024 * 1024);  // 10 MB
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), KVortexError::OutOfMemory);
}

TEST(PinnedHostPool, InvalidDeallocate) {
    auto pool_result = PinnedHostPool::create(1024 * 1024, false);
    ASSERT_TRUE(pool_result.has_value());
    auto pool = std::move(*pool_result);

    // Try to deallocate invalid pointer
    void* fake_ptr = reinterpret_cast<void*>(0x1234);
    auto result = pool->deallocate(fake_ptr);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), KVortexError::InvalidPointer);
}

// ============================================================================
// GPU Async Pool Tests
// ============================================================================

TEST(GPUAsyncPool, CreatePool) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    constexpr size_t pool_size = 100 * 1024 * 1024;  // 100 MB
    auto result = GPUAsyncPool::create(pool_size, stream, 0);

    if (result.has_value()) {
        auto pool = std::move(*result);
        ASSERT_NE(pool, nullptr);
        EXPECT_EQ(pool->total_size(), pool_size);
    } else {
        // CUDA may not be available in test environment
        GTEST_SKIP() << "CUDA not available or GPU pool creation failed";
    }

    cudaStreamDestroy(stream);
}

TEST(GPUAsyncPool, AllocateAndDeallocate) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto pool_result = GPUAsyncPool::create(100 * 1024 * 1024, stream, 0);
    if (!pool_result.has_value()) {
        cudaStreamDestroy(stream);
        GTEST_SKIP() << "CUDA not available";
    }

    auto pool = std::move(*pool_result);

    // Allocate 10 MB
    auto ptr_result = pool->allocate_async(10 * 1024 * 1024);
    ASSERT_TRUE(ptr_result.has_value());
    void* ptr = *ptr_result;
    ASSERT_NE(ptr, nullptr);

    // Sync stream before deallocating
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Deallocate
    EXPECT_TRUE(pool->deallocate_async(ptr).has_value());

    // Sync again
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaStreamDestroy(stream);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
