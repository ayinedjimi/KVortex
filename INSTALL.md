# Installation Guide - KVortex

## Prérequis

### Matériel
- GPU NVIDIA avec Compute Capability 8.6+ (RTX 3090 ou supérieur)
- 16GB+ RAM système
- 50GB+ espace disque

### Logiciels Requis
- CUDA Toolkit 13.0+ (`nvcc` dans le PATH)
- GCC 13.0+ avec support C++23
- CMake 3.28+
- OpenSSL 3.0+

### Logiciels Optionnels
- libnuma (pour NUMA awareness)
- pybind11 (pour bindings Python)
- PyTorch (pour intégration vLLM)
- Google Test (téléchargé automatiquement si absent)

## Installation

### 1. Vérifier l'environnement

```bash
# Vérifier CUDA
nvcc --version

# Vérifier GCC
g++ --version  # Doit être ≥13.0

# Vérifier CMake
cmake --version  # Doit être ≥3.28

# Vérifier GPU
nvidia-smi
```

### 2. Cloner le projet

```bash
cd /votre/repertoire
# Le projet est déjà à /home/deeptechadmin/kvortex
```

### 3. Build

```bash
cd kvortex

# Configuration
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
    -DKVORTEX_BUILD_TESTS=ON

# Compilation
cmake --build build -j$(nproc)
```

### 4. Tests

```bash
# Exécuter tous les tests
ctest --test-dir build --output-on-failure

# Ou individuellement
./build/test_memory
./build/test_integration
```

## Utilisation

### Exemple C++

```cpp
#include "kvortex/api/kvortex.hpp"

int main() {
    // Configuration
    kvortex::KVortexConfig config;
    config.cpu_pool_size_bytes = 16ULL * 1024 * 1024 * 1024;  // 16GB
    config.gpu_pool_size_bytes = 8ULL * 1024 * 1024 * 1024;   // 8GB
    config.num_transfer_streams = 3;
    config.enable_numa = true;

    // Créer le moteur
    auto engine_result = kvortex::KVortexEngine::create(config);
    if (!engine_result) {
        std::cerr << "Failed to create engine\n";
        return 1;
    }

    auto engine = std::move(*engine_result);

    // Utiliser...
    auto stats = engine->get_stats();
    std::cout << "Cached blocks: " << stats.num_cached_blocks << "\n";
    std::cout << "Hit rate: " << stats.cache_hit_rate << "\n";

    // Nettoyer
    engine->shutdown();
    return 0;
}
```

### Compiler avec KVortex

```bash
g++ -std=c++23 votre_programme.cpp \
    -I/path/to/kvortex/include \
    -L/path/to/kvortex/build \
    -lkvortex_core \
    -lcudart \
    -lssl -lcrypto \
    -lpthread \
    -o votre_programme
```

## Configuration

### Variables d'environnement

```bash
# Spécifier le device CUDA
export CUDA_VISIBLE_DEVICES=0

# Log level
export KVORTEX_LOG_LEVEL=INFO  # TRACE, DEBUG, INFO, WARN, ERROR
```

### Fichier de configuration

```yaml
# kvortex_config.yaml
cpu_pool_size_bytes: 17179869184  # 16GB
gpu_pool_size_bytes: 8589934592   # 8GB
num_transfer_streams: 3
thread_pool_size: 8
chunk_size: 256
eviction_watermark: 0.8
eviction_ratio: 0.2
enable_numa: true
log_level: INFO
```

## Troubleshooting

### Erreur: CUDA not found
```bash
# Ajouter CUDA au PATH
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
```

### Erreur: Out of pinned memory
```bash
# Réduire la taille du pool CPU
config.cpu_pool_size_bytes = 8ULL * 1024 * 1024 * 1024;  // 8GB au lieu de 16GB
```

### Erreur: Tests échouent
```bash
# Vérifier les logs
./build/test_memory --gtest_output=xml:test_results.xml

# Activer debug logging
export KVORTEX_LOG_LEVEL=DEBUG
```

## Support

- Documentation: README.md, COMPLETE.md
- Tests: tests/ directory
- Exemples: tests/test_integration.cpp

## Licence

Apache 2.0 - Voir LICENSE file
