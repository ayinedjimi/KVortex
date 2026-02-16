# 📖 KVortex - Guide d'Utilisation Complet / Complete Usage Guide

**Version 1.0** | [English](#english) | [Français](#français)

---

## <a id="english"></a>🇬🇧 English - Complete Usage Guide

### Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Integration with vLLM](#integration-with-vllm)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring and Statistics](#monitoring-and-statistics)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)

---

### Quick Start

The fastest way to get started with KVortex:

```bash
# 1. Clone and build
git clone https://github.com/ayinedjimi/KVortex.git
cd KVortex
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Run tests to verify installation
ctest --test-dir build --output-on-failure

# 3. Use the library
# See examples below
```

---

### Installation

#### System Requirements

**Minimum:**
- GPU: NVIDIA RTX 3090 (24GB VRAM, Compute Capability 8.6)
- RAM: 32GB system memory
- Storage: 50GB free space
- OS: Ubuntu 22.04 LTS or newer

**Recommended:**
- GPU: NVIDIA RTX 4090 or A100
- RAM: 64GB+ system memory
- Storage: 100GB+ NVMe SSD
- OS: Ubuntu 24.04 LTS

#### Software Dependencies

**Required:**
```bash
# CUDA Toolkit
sudo apt install cuda-toolkit-13-1

# GCC with C++23 support
sudo apt install g++-13

# CMake
sudo apt install cmake

# OpenSSL
sudo apt install libssl-dev
```

**Optional:**
```bash
# NUMA support (recommended for multi-socket systems)
sudo apt install libnuma-dev

# Python bindings
sudo apt install python3-dev pybind11-dev

# vLLM integration
pip install vllm>=0.15.0
```

#### Build from Source

**Standard Build:**
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
    -DKVORTEX_BUILD_TESTS=ON

cmake --build build -j$(nproc)
```

**Debug Build (for development):**
```bash
cmake -B build-debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DKVORTEX_ENABLE_ASAN=ON \
    -DKVORTEX_ENABLE_UBSAN=ON

cmake --build build-debug -j$(nproc)
```

**Install System-Wide:**
```bash
sudo cmake --install build --prefix /usr/local
```

#### Using Pre-Built Binaries

Download from [GitHub Releases](https://github.com/ayinedjimi/KVortex/releases/latest):

```bash
# Download library
wget https://github.com/ayinedjimi/KVortex/releases/download/v1.0/kvortex-v1.0-linux-x86_64-cuda13.1.tar.gz

# Download headers
wget https://github.com/ayinedjimi/KVortex/releases/download/v1.0/kvortex-v1.0-headers.tar.gz

# Extract
tar -xzf kvortex-v1.0-linux-x86_64-cuda13.1.tar.gz
tar -xzf kvortex-v1.0-headers.tar.gz

# Install
sudo cp libkvortex_core.a /usr/local/lib/
sudo cp -r include/kvortex /usr/local/include/
```

---

### Basic Usage

#### Example 1: Simple Cache Operations

```cpp
#include "kvortex/api/kvortex.hpp"
#include <iostream>
#include <vector>

int main() {
    // Configure KVortex
    kvortex::KVortexConfig config;
    config.cpu_pool_size_bytes = 4ULL * 1024 * 1024 * 1024;  // 4GB
    config.gpu_pool_size_bytes = 2ULL * 1024 * 1024 * 1024;  // 2GB
    config.num_transfer_streams = 2;
    config.enable_numa = false;  // Disable for single-socket systems

    // Create engine
    auto engine_result = kvortex::KVortexEngine::create(config);
    if (!engine_result) {
        std::cerr << "Failed to create KVortex engine\n";
        return 1;
    }
    auto engine = std::move(*engine_result);

    // Prepare test data
    const size_t block_size = 1024 * 1024;  // 1MB blocks
    std::vector<uint8_t> data1(block_size, 0xAA);
    std::vector<uint8_t> data2(block_size, 0xBB);

    // Hash blocks to get IDs
    kvortex::BlockHasher hasher;
    auto id1 = hasher.hash_data(data1.data(), data1.size());
    auto id2 = hasher.hash_data(data2.data(), data2.size());

    // Save blocks
    std::vector<kvortex::BlockID> block_ids = {id1, id2};
    std::vector<const void*> data_ptrs = {data1.data(), data2.data()};
    std::vector<size_t> sizes = {block_size, block_size};

    auto save_result = engine->save_blocks(block_ids, data_ptrs, sizes);
    if (!save_result) {
        std::cerr << "Failed to save blocks\n";
        return 1;
    }
    std::cout << "✓ Saved 2 blocks to cache\n";

    // Check if blocks are cached
    auto cached = engine->check_blocks(block_ids);
    std::cout << "Block 1 cached: " << (cached[0] ? "Yes" : "No") << "\n";
    std::cout << "Block 2 cached: " << (cached[1] ? "Yes" : "No") << "\n";

    // Load blocks back
    std::vector<uint8_t> loaded1(block_size);
    std::vector<uint8_t> loaded2(block_size);
    std::vector<void*> load_ptrs = {loaded1.data(), loaded2.data()};

    auto load_result = engine->load_blocks(block_ids, load_ptrs, sizes);
    if (!load_result) {
        std::cerr << "Failed to load blocks\n";
        return 1;
    }
    std::cout << "✓ Loaded 2 blocks from cache\n";

    // Verify data integrity
    bool match1 = (loaded1 == data1);
    bool match2 = (loaded2 == data2);
    std::cout << "Data integrity: " << (match1 && match2 ? "PASS" : "FAIL") << "\n";

    // Get statistics
    auto stats = engine->get_stats();
    std::cout << "\nCache Statistics:\n";
    std::cout << "  Cached blocks: " << stats.num_cached_blocks << "\n";
    std::cout << "  Hit rate: " << (stats.cache_hit_rate * 100) << "%\n";
    std::cout << "  Memory used: " << (stats.memory_used_bytes / (1024*1024)) << " MB\n";

    // Cleanup
    engine->shutdown();
    return 0;
}
```

**Compile:**
```bash
g++ -std=c++23 example.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lkvortex_core \
    -lcudart \
    -lssl -lcrypto \
    -lpthread \
    -o example

./example
```

#### Example 2: Async Operations

```cpp
#include "kvortex/api/kvortex.hpp"
#include <iostream>
#include <chrono>

int main() {
    kvortex::KVortexConfig config;
    config.cpu_pool_size_bytes = 8ULL * 1024 * 1024 * 1024;  // 8GB
    config.num_transfer_streams = 3;  // Use 3 streams for async

    auto engine = kvortex::KVortexEngine::create(config).value();

    // Prepare large blocks
    const size_t block_size = 10 * 1024 * 1024;  // 10MB
    std::vector<uint8_t> large_block(block_size);
    kvortex::BlockHasher hasher;
    auto block_id = hasher.hash_data(large_block.data(), large_block.size());

    // Async save
    std::vector<kvortex::BlockID> ids = {block_id};
    std::vector<const void*> ptrs = {large_block.data()};
    std::vector<size_t> sizes = {block_size};

    auto start = std::chrono::steady_clock::now();

    auto handle_result = engine->save_blocks_async(ids, ptrs, sizes);
    if (!handle_result) {
        std::cerr << "Failed to start async save\n";
        return 1;
    }
    auto handle = *handle_result;

    std::cout << "Async save initiated...\n";

    // Do other work while transfer happens
    std::cout << "Doing other work...\n";

    // Wait for completion
    auto wait_result = engine->wait(handle);
    if (!wait_result) {
        std::cerr << "Async save failed\n";
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "✓ Async save completed in " << duration.count() << " ms\n";
    std::cout << "Bandwidth: " << (block_size / (1024.0 * 1024.0) / (duration.count() / 1000.0))
              << " MB/s\n";

    engine->shutdown();
    return 0;
}
```

---

### Advanced Configuration

#### Configuration Options

```cpp
struct KVortexConfig {
    // Memory pools
    size_t cpu_pool_size_bytes = 16ULL * 1024 * 1024 * 1024;  // 16GB default
    size_t gpu_pool_size_bytes = 8ULL * 1024 * 1024 * 1024;   // 8GB default

    // Transfer settings
    int num_transfer_streams = 3;        // 2-4 recommended
    size_t transfer_chunk_size = 256;    // Tokens per chunk

    // Threading
    int thread_pool_size = 8;            // Worker threads

    // Cache management
    float eviction_watermark = 0.8f;     // Evict at 80% full
    float eviction_ratio = 0.2f;         // Evict 20% when triggered

    // NUMA awareness
    bool enable_numa = true;             // Enable for multi-socket

    // Logging
    std::string log_level = "INFO";      // TRACE, DEBUG, INFO, WARN, ERROR
};
```

#### Tuning for Different Workloads

**Small Models (7B-13B parameters):**
```cpp
config.cpu_pool_size_bytes = 8ULL * 1024 * 1024 * 1024;   // 8GB
config.gpu_pool_size_bytes = 4ULL * 1024 * 1024 * 1024;   // 4GB
config.num_transfer_streams = 2;
config.eviction_watermark = 0.9f;  // Less aggressive eviction
```

**Large Models (70B+ parameters):**
```cpp
config.cpu_pool_size_bytes = 64ULL * 1024 * 1024 * 1024;  // 64GB
config.gpu_pool_size_bytes = 20ULL * 1024 * 1024 * 1024;  // 20GB
config.num_transfer_streams = 4;
config.eviction_watermark = 0.75f;  // More aggressive eviction
config.eviction_ratio = 0.3f;
```

**High Throughput (many concurrent requests):**
```cpp
config.thread_pool_size = 16;  // More workers
config.num_transfer_streams = 4;
config.transfer_chunk_size = 512;  // Larger chunks
```

---

### Integration with vLLM

#### Python Example

```python
import kvortex_cpp
from vllm import LLM, SamplingParams

# Configure KVortex
config = kvortex_cpp.KVortexConfig()
config.cpu_pool_size_bytes = 32 * 1024**3  # 32GB
config.gpu_pool_size_bytes = 16 * 1024**3  # 16GB
config.num_transfer_streams = 3
config.enable_numa = True

# Create KVortex engine
kvortex_engine = kvortex_cpp.KVortexEngine.create(config)
print("✓ KVortex engine created")

# Initialize vLLM with KVortex backend
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    kv_cache_backend="kvortex",
    kv_connector=kvortex_engine,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)
print("✓ vLLM initialized with KVortex")

# Define prompts
prompts = [
    "Explain quantum computing in simple terms:",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis:"
]

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# Generate with automatic KV cache offloading
print("\nGenerating responses...")
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"\n--- Prompt {i+1} ---")
    print(output.outputs[0].text)

# Get KVortex statistics
stats = kvortex_engine.get_stats()
print(f"\n📊 Cache Statistics:")
print(f"  Cached blocks: {stats.num_cached_blocks}")
print(f"  Cache hit rate: {stats.cache_hit_rate * 100:.2f}%")
print(f"  Memory used: {stats.memory_used_bytes / (1024**3):.2f} GB")
print(f"  Evictions: {stats.num_evictions}")

# Cleanup
kvortex_engine.shutdown()
```

#### Performance Comparison

```python
import time
import kvortex_cpp
from vllm import LLM, SamplingParams

def benchmark_with_kvortex():
    config = kvortex_cpp.KVortexConfig()
    config.cpu_pool_size_bytes = 32 * 1024**3
    engine = kvortex_cpp.KVortexEngine.create(config)

    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        kv_cache_backend="kvortex",
        kv_connector=engine
    )

    prompts = ["Test prompt"] * 100
    sampling_params = SamplingParams(max_tokens=128)

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    duration = time.time() - start

    stats = engine.get_stats()
    engine.shutdown()

    return duration, stats

def benchmark_without_kvortex():
    llm = LLM(model="meta-llama/Llama-2-70b-hf")

    prompts = ["Test prompt"] * 100
    sampling_params = SamplingParams(max_tokens=128)

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    duration = time.time() - start

    return duration, None

# Run benchmarks
print("Benchmarking WITH KVortex...")
with_time, with_stats = benchmark_with_kvortex()

print("Benchmarking WITHOUT KVortex...")
without_time, _ = benchmark_without_kvortex()

# Results
print(f"\n📊 Benchmark Results (100 requests):")
print(f"  Without KVortex: {without_time:.2f}s")
print(f"  With KVortex:    {with_time:.2f}s")
print(f"  Speedup:         {without_time / with_time:.2f}x")
print(f"\n  Cache hit rate:  {with_stats.cache_hit_rate * 100:.2f}%")
```

---

### Performance Tuning

#### 1. Memory Pool Sizing

**Rule of Thumb:**
- CPU pool: 2-4x your model's KV cache size
- GPU pool: 0.5-1x model's KV cache size

**Check optimal size:**
```cpp
auto stats = engine->get_stats();
float usage = (float)stats.memory_used_bytes / config.cpu_pool_size_bytes;

if (usage > 0.9) {
    std::cout << "⚠️  Pool too small, increase size\n";
} else if (usage < 0.3) {
    std::cout << "ℹ️  Pool oversized, can reduce\n";
} else {
    std::cout << "✓ Pool size optimal\n";
}
```

#### 2. Stream Count Tuning

```bash
# Benchmark different stream counts
for streams in 1 2 3 4 5; do
    echo "Testing with $streams streams..."
    # Run your workload with config.num_transfer_streams = $streams
    # Measure throughput
done
```

**Typical results:**
- 1 stream: ~12 GB/s
- 2 streams: ~18 GB/s
- 3 streams: ~22 GB/s (optimal for RTX 3090)
- 4 streams: ~23 GB/s (diminishing returns)
- 5+ streams: ~23 GB/s (no improvement)

#### 3. NUMA Awareness

**Check NUMA topology:**
```bash
numactl --hardware
```

**Enable NUMA if you have multiple sockets:**
```cpp
config.enable_numa = true;  // For dual-socket systems
```

**Bind to specific NUMA node:**
```bash
numactl --cpunodebind=0 --membind=0 ./your_program
```

#### 4. Eviction Policy Tuning

```cpp
// Conservative (fewer evictions, higher memory usage)
config.eviction_watermark = 0.95f;
config.eviction_ratio = 0.1f;

// Aggressive (more evictions, lower memory usage)
config.eviction_watermark = 0.7f;
config.eviction_ratio = 0.3f;

// Balanced (default)
config.eviction_watermark = 0.8f;
config.eviction_ratio = 0.2f;
```

---

### Monitoring and Statistics

#### Real-Time Monitoring

```cpp
#include "kvortex/api/kvortex.hpp"
#include <iostream>
#include <thread>
#include <chrono>

void monitor_cache(kvortex::KVortexEngine* engine, int interval_seconds) {
    while (true) {
        auto stats = engine->get_stats();

        std::cout << "\033[2J\033[H";  // Clear screen
        std::cout << "=== KVortex Real-Time Statistics ===\n\n";

        std::cout << "📦 Cache:\n";
        std::cout << "  Blocks cached:  " << stats.num_cached_blocks << "\n";
        std::cout << "  Hit rate:       " << (stats.cache_hit_rate * 100) << "%\n";
        std::cout << "  Total hits:     " << stats.cache_hits << "\n";
        std::cout << "  Total misses:   " << stats.cache_misses << "\n";

        std::cout << "\n💾 Memory:\n";
        float mem_gb = stats.memory_used_bytes / (1024.0 * 1024.0 * 1024.0);
        float mem_percent = (stats.memory_used_bytes * 100.0) / stats.memory_capacity_bytes;
        std::cout << "  Used:           " << mem_gb << " GB (" << mem_percent << "%)\n";
        std::cout << "  Capacity:       " << (stats.memory_capacity_bytes / (1024.0*1024*1024)) << " GB\n";

        std::cout << "\n🔄 Evictions:\n";
        std::cout << "  Total evictions: " << stats.num_evictions << "\n";
        std::cout << "  Evicted blocks:  " << stats.evicted_blocks << "\n";

        std::cout << "\n⚡ Performance:\n";
        std::cout << "  Avg latency:     " << stats.avg_latency_ms << " ms\n";
        std::cout << "  Transfer rate:   " << stats.transfer_rate_mbps << " MB/s\n";

        std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
    }
}

int main() {
    auto config = kvortex::KVortexConfig{};
    auto engine = kvortex::KVortexEngine::create(config).value();

    // Start monitoring in background thread
    std::thread monitor_thread(monitor_cache, engine.get(), 2);

    // Your workload here...

    monitor_thread.join();
    engine->shutdown();
    return 0;
}
```

#### Logging Configuration

```cpp
// Set log level via environment variable
setenv("KVORTEX_LOG_LEVEL", "DEBUG", 1);

// Or in config
config.log_level = "TRACE";  // Most verbose
config.log_level = "DEBUG";  // Detailed info
config.log_level = "INFO";   // Default
config.log_level = "WARN";   // Warnings only
config.log_level = "ERROR";  // Errors only
```

---

### Troubleshooting

#### Common Issues

**1. Out of Memory Errors**

```
Error: Failed to allocate pinned memory
```

**Solution:**
```bash
# Check available memory
free -h

# Reduce pool size
config.cpu_pool_size_bytes = 4ULL * 1024 * 1024 * 1024;  # Try 4GB

# Check system limits
ulimit -l  # If not "unlimited", increase it
```

**2. CUDA Errors**

```
Error: CUDA error 2: out of memory
```

**Solution:**
```bash
# Check GPU memory
nvidia-smi

# Reduce GPU pool size
config.gpu_pool_size_bytes = 2ULL * 1024 * 1024 * 1024;  # Try 2GB

# Check CUDA installation
nvcc --version
```

**3. Slow Performance**

```
Warning: Transfer rate below 10 GB/s
```

**Solution:**
```cpp
// Increase stream count
config.num_transfer_streams = 4;

// Check PCIe bandwidth
// Run: nvidia-smi topo -m

// Verify NUMA configuration
config.enable_numa = true;
```

**4. Compilation Errors**

```
error: 'std::expected' is not a member of 'std'
```

**Solution:**
```bash
# Verify GCC version (need 13.3+)
g++ --version

# Use newer GCC
sudo apt install g++-13
export CXX=g++-13
```

#### Debug Mode

```bash
# Build with debug symbols
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

# Run with sanitizers
./build-debug/your_program

# Check for memory leaks
valgrind --leak-check=full ./your_program
```

---

### API Reference

#### Core Classes

**KVortexEngine**
```cpp
class KVortexEngine {
public:
    // Factory method
    static Result<std::unique_ptr<KVortexEngine>> create(const KVortexConfig& config);

    // Synchronous operations
    VoidResult save_blocks(const std::vector<BlockID>& ids,
                          const std::vector<const void*>& data,
                          const std::vector<size_t>& sizes);

    VoidResult load_blocks(const std::vector<BlockID>& ids,
                          const std::vector<void*>& buffers,
                          const std::vector<size_t>& sizes);

    std::vector<bool> check_blocks(const std::vector<BlockID>& ids) const;

    // Asynchronous operations
    Result<AsyncHandle> save_blocks_async(/* ... */);
    VoidResult wait(AsyncHandle handle);

    // Statistics
    CacheStats get_stats() const;

    // Shutdown
    void shutdown();
};
```

**BlockHasher**
```cpp
class BlockHasher {
public:
    // Hash raw data
    BlockID hash_data(const void* data, size_t size);

    // Hash tokens
    BlockID hash_tokens(const std::vector<int32_t>& tokens);

    // Hash in chunks
    std::vector<BlockID> hash_chunks(const std::vector<int32_t>& tokens,
                                     size_t chunk_size);
};
```

**CacheStats**
```cpp
struct CacheStats {
    size_t num_cached_blocks;
    size_t memory_used_bytes;
    size_t memory_capacity_bytes;
    double cache_hit_rate;
    size_t cache_hits;
    size_t cache_misses;
    size_t num_evictions;
    size_t evicted_blocks;
    double avg_latency_ms;
    double transfer_rate_mbps;
};
```

---

### Best Practices

#### 1. Resource Management

```cpp
// ✓ Good: Use RAII
{
    auto engine = KVortexEngine::create(config).value();
    // Use engine...
    engine->shutdown();
}  // Automatic cleanup

// ✗ Bad: Manual management
KVortexEngine* engine = new KVortexEngine();
// Use engine...
delete engine;  // Error-prone
```

#### 2. Error Handling

```cpp
// ✓ Good: Check all results
auto result = engine->save_blocks(ids, ptrs, sizes);
if (!result) {
    std::cerr << "Save failed: " << to_string(result.error()) << "\n";
    return 1;
}

// ✗ Bad: Ignore errors
engine->save_blocks(ids, ptrs, sizes);  // May fail silently
```

#### 3. Batching

```cpp
// ✓ Good: Batch operations
std::vector<BlockID> ids(100);
std::vector<const void*> ptrs(100);
std::vector<size_t> sizes(100);
engine->save_blocks(ids, ptrs, sizes);  // One call

// ✗ Bad: Individual operations
for (int i = 0; i < 100; i++) {
    engine->save_blocks({ids[i]}, {ptrs[i]}, {sizes[i]});  // 100 calls
}
```

#### 4. Memory Pooling

```cpp
// ✓ Good: Pre-allocate buffers
std::vector<uint8_t> buffer(block_size);
for (int i = 0; i < 1000; i++) {
    // Reuse buffer
    engine->load_blocks({ids[i]}, {buffer.data()}, {block_size});
}

// ✗ Bad: Allocate every time
for (int i = 0; i < 1000; i++) {
    std::vector<uint8_t> buffer(block_size);  // Expensive!
    engine->load_blocks({ids[i]}, {buffer.data()}, {block_size});
}
```

#### 5. Monitoring

```cpp
// ✓ Good: Regular statistics checks
auto stats = engine->get_stats();
if (stats.cache_hit_rate < 0.5) {
    std::cout << "⚠️  Low hit rate, consider tuning\n";
}

// ✓ Good: Monitor evictions
if (stats.num_evictions > expected_evictions * 2) {
    std::cout << "⚠️  Excessive evictions, increase pool size\n";
}
```

---

## <a id="français"></a>🇫🇷 Français - Guide d'Utilisation Complet

### Table des Matières

1. [Démarrage Rapide](#démarrage-rapide-fr)
2. [Installation](#installation-fr)
3. [Utilisation Basique](#utilisation-basique-fr)
4. [Configuration Avancée](#configuration-avancée-fr)
5. [Intégration avec vLLM](#intégration-avec-vllm-fr)
6. [Optimisation des Performances](#optimisation-des-performances-fr)
7. [Surveillance et Statistiques](#surveillance-et-statistiques-fr)
8. [Résolution de Problèmes](#résolution-de-problèmes-fr)
9. [Référence API](#référence-api-fr)
10. [Bonnes Pratiques](#bonnes-pratiques-fr)

---

### <a id="démarrage-rapide-fr"></a>Démarrage Rapide

La façon la plus rapide de démarrer avec KVortex :

```bash
# 1. Cloner et compiler
git clone https://github.com/ayinedjimi/KVortex.git
cd KVortex
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Lancer les tests pour vérifier l'installation
ctest --test-dir build --output-on-failure

# 3. Utiliser la bibliothèque
# Voir les exemples ci-dessous
```

---

### <a id="installation-fr"></a>Installation

#### Configuration Système Requise

**Minimum :**
- GPU : NVIDIA RTX 3090 (24GB VRAM, Compute Capability 8.6)
- RAM : 32GB mémoire système
- Stockage : 50GB espace libre
- OS : Ubuntu 22.04 LTS ou plus récent

**Recommandé :**
- GPU : NVIDIA RTX 4090 ou A100
- RAM : 64GB+ mémoire système
- Stockage : 100GB+ SSD NVMe
- OS : Ubuntu 24.04 LTS

#### Dépendances Logicielles

**Requis :**
```bash
# CUDA Toolkit
sudo apt install cuda-toolkit-13-1

# GCC avec support C++23
sudo apt install g++-13

# CMake
sudo apt install cmake

# OpenSSL
sudo apt install libssl-dev
```

**Optionnel :**
```bash
# Support NUMA (recommandé pour systèmes multi-socket)
sudo apt install libnuma-dev

# Bindings Python
sudo apt install python3-dev pybind11-dev

# Intégration vLLM
pip install vllm>=0.15.0
```

#### Compilation depuis les Sources

**Build Standard :**
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
    -DKVORTEX_BUILD_TESTS=ON

cmake --build build -j$(nproc)
```

**Build Debug (pour développement) :**
```bash
cmake -B build-debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DKVORTEX_ENABLE_ASAN=ON \
    -DKVORTEX_ENABLE_UBSAN=ON

cmake --build build-debug -j$(nproc)
```

**Installation Système :**
```bash
sudo cmake --install build --prefix /usr/local
```

#### Utilisation des Binaires Pré-compilés

Télécharger depuis [GitHub Releases](https://github.com/ayinedjimi/KVortex/releases/latest) :

```bash
# Télécharger la bibliothèque
wget https://github.com/ayinedjimi/KVortex/releases/download/v1.0/kvortex-v1.0-linux-x86_64-cuda13.1.tar.gz

# Télécharger les headers
wget https://github.com/ayinedjimi/KVortex/releases/download/v1.0/kvortex-v1.0-headers.tar.gz

# Extraire
tar -xzf kvortex-v1.0-linux-x86_64-cuda13.1.tar.gz
tar -xzf kvortex-v1.0-headers.tar.gz

# Installer
sudo cp libkvortex_core.a /usr/local/lib/
sudo cp -r include/kvortex /usr/local/include/
```

---

### <a id="utilisation-basique-fr"></a>Utilisation Basique

#### Exemple 1 : Opérations de Cache Simples

```cpp
#include "kvortex/api/kvortex.hpp"
#include <iostream>
#include <vector>

int main() {
    // Configurer KVortex
    kvortex::KVortexConfig config;
    config.cpu_pool_size_bytes = 4ULL * 1024 * 1024 * 1024;  // 4GB
    config.gpu_pool_size_bytes = 2ULL * 1024 * 1024 * 1024;  // 2GB
    config.num_transfer_streams = 2;
    config.enable_numa = false;  // Désactiver pour systèmes mono-socket

    // Créer le moteur
    auto engine_result = kvortex::KVortexEngine::create(config);
    if (!engine_result) {
        std::cerr << "Échec création moteur KVortex\n";
        return 1;
    }
    auto engine = std::move(*engine_result);

    // Préparer des données de test
    const size_t block_size = 1024 * 1024;  // Blocs de 1MB
    std::vector<uint8_t> data1(block_size, 0xAA);
    std::vector<uint8_t> data2(block_size, 0xBB);

    // Hacher les blocs pour obtenir les IDs
    kvortex::BlockHasher hasher;
    auto id1 = hasher.hash_data(data1.data(), data1.size());
    auto id2 = hasher.hash_data(data2.data(), data2.size());

    // Sauvegarder les blocs
    std::vector<kvortex::BlockID> block_ids = {id1, id2};
    std::vector<const void*> data_ptrs = {data1.data(), data2.data()};
    std::vector<size_t> sizes = {block_size, block_size};

    auto save_result = engine->save_blocks(block_ids, data_ptrs, sizes);
    if (!save_result) {
        std::cerr << "Échec sauvegarde blocs\n";
        return 1;
    }
    std::cout << "✓ 2 blocs sauvegardés dans le cache\n";

    // Vérifier si les blocs sont en cache
    auto cached = engine->check_blocks(block_ids);
    std::cout << "Bloc 1 en cache : " << (cached[0] ? "Oui" : "Non") << "\n";
    std::cout << "Bloc 2 en cache : " << (cached[1] ? "Oui" : "Non") << "\n";

    // Charger les blocs
    std::vector<uint8_t> loaded1(block_size);
    std::vector<uint8_t> loaded2(block_size);
    std::vector<void*> load_ptrs = {loaded1.data(), loaded2.data()};

    auto load_result = engine->load_blocks(block_ids, load_ptrs, sizes);
    if (!load_result) {
        std::cerr << "Échec chargement blocs\n";
        return 1;
    }
    std::cout << "✓ 2 blocs chargés depuis le cache\n";

    // Vérifier l'intégrité des données
    bool match1 = (loaded1 == data1);
    bool match2 = (loaded2 == data2);
    std::cout << "Intégrité données : " << (match1 && match2 ? "OK" : "ÉCHEC") << "\n";

    // Obtenir les statistiques
    auto stats = engine->get_stats();
    std::cout << "\nStatistiques Cache :\n";
    std::cout << "  Blocs en cache : " << stats.num_cached_blocks << "\n";
    std::cout << "  Taux de hit : " << (stats.cache_hit_rate * 100) << "%\n";
    std::cout << "  Mémoire utilisée : " << (stats.memory_used_bytes / (1024*1024)) << " MB\n";

    // Nettoyage
    engine->shutdown();
    return 0;
}
```

**Compiler :**
```bash
g++ -std=c++23 exemple.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lkvortex_core \
    -lcudart \
    -lssl -lcrypto \
    -lpthread \
    -o exemple

./exemple
```

---

### 📞 Support et Ressources

**Documentation Complète :**
- [GitHub Repository](https://github.com/ayinedjimi/KVortex)
- [Installation Guide](INSTALL.md)
- [API Reference](https://github.com/ayinedjimi/KVortex/wiki)

**Support Professionnel :**
- 📧 Email : contact@ayinedjimi-consultants.fr
- 🌐 Site : [ayinedjimi-consultants.fr](https://ayinedjimi-consultants.fr)
- 📝 Blog : [Articles IA/ML](https://ayinedjimi-consultants.fr/blog/categories/intelligence-artificielle)

**Communauté :**
- 🐙 GitHub Issues : [Report bugs](https://github.com/ayinedjimi/KVortex/issues)
- 💬 Discussions : [GitHub Discussions](https://github.com/ayinedjimi/KVortex/discussions)

---

<div align="center">

**⭐ Si ce guide vous est utile, n'hésitez pas à mettre une étoile au projet ! ⭐**

Made with ❤️ by **Ayi NEDJIMI**

</div>
