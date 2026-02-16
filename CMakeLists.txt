cmake_minimum_required(VERSION 3.28)
project(KVortex VERSION 1.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# Project Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -fsanitize=undefined")

# CUDA architecture (RTX 3090 = compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

# ============================================================================
# Dependencies
# ============================================================================

# CUDA
find_package(CUDAToolkit 13.0 REQUIRED)

# OpenSSL (for SHA256)
find_package(OpenSSL REQUIRED)

# NUMA (optional, Linux only)
find_library(NUMA_LIBRARY numa)
if(NUMA_LIBRARY)
    add_definitions(-DHAVE_NUMA)
    message(STATUS "NUMA support enabled")
else()
    message(STATUS "NUMA support disabled (libnuma not found)")
endif()

# Python and pybind11 (optional, for bindings)
option(KVORTEX_BUILD_PYTHON "Build Python bindings" ON)
if(KVORTEX_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, Python bindings will be built")
    else()
        message(WARNING "pybind11 not found, Python bindings disabled")
        set(KVORTEX_BUILD_PYTHON OFF)
    endif()
endif()

# PyTorch (optional, for tensor integration)
option(KVORTEX_WITH_TORCH "Enable PyTorch integration" OFF)
if(KVORTEX_WITH_TORCH)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "PyTorch found: ${TORCH_VERSION}")
    else()
        message(WARNING "PyTorch not found, tensor integration disabled")
        set(KVORTEX_WITH_TORCH OFF)
    endif()
endif()

# Google Test (for unit tests)
option(KVORTEX_BUILD_TESTS "Build tests" ON)
if(KVORTEX_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# ============================================================================
# Core Library
# ============================================================================

add_library(kvortex_core STATIC
    src/core/types.cpp
    src/memory/pool.cpp
    src/transfer/stream_manager.cpp
    src/cache/index.cpp
    src/cache/eviction.cpp
    src/storage/cpu_backend.cpp
    src/api/kvortex.cpp
)

target_include_directories(kvortex_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvortex_core PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(NUMA_LIBRARY)
    target_link_libraries(kvortex_core PUBLIC ${NUMA_LIBRARY})
endif()

if(KVORTEX_WITH_TORCH)
    target_link_libraries(kvortex_core PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(kvortex_core PUBLIC KVORTEX_WITH_TORCH)
endif()

# ============================================================================
# CUDA Kernels (placeholder for LMCache kernels)
# ============================================================================

# Note: LMCache CUDA kernels will be added in Phase 2
# add_library(kvortex_kernels STATIC
#     kernels/ac_dec.cu
#     kernels/consolidate.cu
# )
#
# set_target_properties(kvortex_kernels PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "86"
# )

# ============================================================================
# Python Bindings (pybind11)
# ============================================================================

if(KVORTEX_BUILD_PYTHON)
    pybind11_add_module(kvortex_cpp
        bindings/bindings.cpp
    )

    target_link_libraries(kvortex_cpp PRIVATE
        kvortex_core
        # kvortex_kernels  # Add when available
    )

    # Install Python module
    install(TARGETS kvortex_cpp DESTINATION .)
endif()

# ============================================================================
# Tests
# ============================================================================

if(KVORTEX_BUILD_TESTS)
    enable_testing()

    # Memory pool tests
    add_executable(test_memory
        tests/test_memory.cpp
    )

    target_link_libraries(test_memory PRIVATE
        kvortex_core
        GTest::gtest_main
    )

    add_test(NAME MemoryPoolTests COMMAND test_memory)

    # Integration tests
    add_executable(test_integration
        tests/test_integration.cpp
    )

    target_link_libraries(test_integration PRIVATE
        kvortex_core
        GTest::gtest_main
    )

    add_test(NAME IntegrationTests COMMAND test_integration)
endif()

# ============================================================================
# Benchmarks
# ============================================================================

option(KVORTEX_BUILD_BENCHMARKS "Build benchmarks" OFF)
if(KVORTEX_BUILD_BENCHMARKS)
    # Placeholder for Google Benchmark integration
    # find_package(benchmark REQUIRED)
    # add_executable(benchmark_transfer benchmarks/transfer_bandwidth.cpp)
    # target_link_libraries(benchmark_transfer PRIVATE kvortex_core benchmark::benchmark_main)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS kvortex_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/kvortex
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "KVortex Configuration:")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:    C++${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard:   C++${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA arch:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python bindings: ${KVORTEX_BUILD_PYTHON}")
message(STATUS "  PyTorch support: ${KVORTEX_WITH_TORCH}")
message(STATUS "  NUMA support:    ${NUMA_LIBRARY}")
message(STATUS "  Build tests:     ${KVORTEX_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${KVORTEX_BUILD_BENCHMARKS}")
message(STATUS "")
