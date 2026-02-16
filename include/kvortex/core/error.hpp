// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <expected>
#include <string>
#include <string_view>

namespace kvortex {

// ============================================================================
// Error Types (for std::expected)
// ============================================================================

enum class KVortexError {
    // Memory errors
    OutOfMemory,
    AllocationFailed,
    InvalidPointer,
    MemoryLeak,

    // CUDA errors
    CUDAError,
    CUDAAllocationFailed,
    CUDAMemcpyFailed,
    CUDAStreamError,
    CUDAEventError,

    // Cache errors
    BlockNotFound,
    BlockAlreadyExists,
    InvalidBlockID,
    EvictionFailed,

    // Storage errors
    IOError,
    FileNotFound,
    DiskFull,
    NetworkError,
    BackendUnavailable,

    // vLLM integration errors
    InvalidTensorShape,
    TensorNotContiguous,
    UnsupportedDataType,
    SlotMappingError,

    // Threading errors
    LockTimeout,
    QueueFull,
    ThreadPoolShutdown,

    // Configuration errors
    InvalidConfiguration,
    MissingRequirement,

    // General errors
    NotImplemented,
    InternalError,
    InvalidArgument,
    OperationCancelled
};

/// Convert error to human-readable string
constexpr std::string_view error_to_string(KVortexError err) noexcept {
    switch (err) {
        // Memory errors
        case KVortexError::OutOfMemory: return "Out of memory";
        case KVortexError::AllocationFailed: return "Memory allocation failed";
        case KVortexError::InvalidPointer: return "Invalid memory pointer";
        case KVortexError::MemoryLeak: return "Memory leak detected";

        // CUDA errors
        case KVortexError::CUDAError: return "CUDA error";
        case KVortexError::CUDAAllocationFailed: return "CUDA allocation failed";
        case KVortexError::CUDAMemcpyFailed: return "CUDA memcpy failed";
        case KVortexError::CUDAStreamError: return "CUDA stream error";
        case KVortexError::CUDAEventError: return "CUDA event error";

        // Cache errors
        case KVortexError::BlockNotFound: return "Block not found in cache";
        case KVortexError::BlockAlreadyExists: return "Block already exists";
        case KVortexError::InvalidBlockID: return "Invalid block identifier";
        case KVortexError::EvictionFailed: return "Cache eviction failed";

        // Storage errors
        case KVortexError::IOError: return "I/O error";
        case KVortexError::FileNotFound: return "File not found";
        case KVortexError::DiskFull: return "Disk full";
        case KVortexError::NetworkError: return "Network error";
        case KVortexError::BackendUnavailable: return "Storage backend unavailable";

        // vLLM integration errors
        case KVortexError::InvalidTensorShape: return "Invalid tensor shape";
        case KVortexError::TensorNotContiguous: return "Tensor is not contiguous";
        case KVortexError::UnsupportedDataType: return "Unsupported data type";
        case KVortexError::SlotMappingError: return "Slot mapping error";

        // Threading errors
        case KVortexError::LockTimeout: return "Lock acquisition timeout";
        case KVortexError::QueueFull: return "Queue is full";
        case KVortexError::ThreadPoolShutdown: return "Thread pool is shut down";

        // Configuration errors
        case KVortexError::InvalidConfiguration: return "Invalid configuration";
        case KVortexError::MissingRequirement: return "Missing requirement";

        // General errors
        case KVortexError::NotImplemented: return "Not implemented";
        case KVortexError::InternalError: return "Internal error";
        case KVortexError::InvalidArgument: return "Invalid argument";
        case KVortexError::OperationCancelled: return "Operation cancelled";

        default: return "Unknown error";
    }
}

// ============================================================================
// Result Type (std::expected alias)
// ============================================================================

template <typename T>
using Result = std::expected<T, KVortexError>;

/// Helper for void results
using VoidResult = Result<void>;

// ============================================================================
// Error Handling Macros
// ============================================================================

/// Return error if result failed
#define KVORTEX_TRY(expr) \
    do { \
        auto _result = (expr); \
        if (!_result) { \
            return std::unexpected(_result.error()); \
        } \
    } while (0)

/// Return error with custom error type if expression fails
#define KVORTEX_TRY_WITH_ERROR(expr, error_type) \
    do { \
        if (!(expr)) { \
            return std::unexpected(error_type); \
        } \
    } while (0)

/// Assign value or return error
#define KVORTEX_ASSIGN_OR_RETURN(var, expr) \
    auto _tmp_##var = (expr); \
    if (!_tmp_##var) { \
        return std::unexpected(_tmp_##var.error()); \
    } \
    var = std::move(*_tmp_##var)

} // namespace kvortex
