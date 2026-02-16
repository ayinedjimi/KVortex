# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in KVortex, please report it responsibly:

### Private Disclosure

Please **DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, contact us directly at:
- **Email:** contact@ayinedjimi-consultants.fr
- **Subject:** [SECURITY] KVortex Vulnerability Report

### What to Include

When reporting a vulnerability, please include:
1. **Description:** Clear description of the vulnerability
2. **Impact:** Potential impact and attack scenarios
3. **Reproduction:** Step-by-step instructions to reproduce
4. **Environment:** System details (OS, CUDA version, GPU model)
5. **Suggested Fix:** If you have one (optional)

### Response Time

- **Initial Response:** Within 48 hours
- **Status Update:** Within 7 days
- **Fix Timeline:** Depends on severity (Critical: 7-14 days, High: 30 days, Medium/Low: 60 days)

### Recognition

We appreciate responsible disclosure and will:
- Credit you in the security advisory (unless you prefer to remain anonymous)
- Keep you updated on the fix progress
- Notify you when the fix is released

## Security Best Practices

When using KVortex in production:

1. **Memory Safety**
   - Always validate input sizes before allocations
   - Use provided error handling (`std::expected`)
   - Monitor memory usage with `get_stats()`

2. **Thread Safety**
   - KVortex is thread-safe by design
   - Follow API documentation for concurrent usage
   - Avoid mixing sync/async operations on same blocks

3. **CUDA Security**
   - Keep CUDA drivers updated
   - Use latest CUDA toolkit (13.1+)
   - Monitor GPU memory limits

4. **Dependencies**
   - Keep OpenSSL updated (3.0+)
   - Use latest stable GCC (13.3+)
   - Regularly update system libraries

5. **Production Deployment**
   - Enable all compiler warnings (`-Wall -Wextra -Werror`)
   - Run memory leak detection before deployment
   - Use sanitizers during testing (ASan, TSan, UBSan)
   - Monitor cache statistics for anomalies

## Known Security Considerations

### Memory Allocations
- Large allocations may fail on systems with limited RAM
- Pinned memory is limited by system configuration
- Always check return values from `create()` and `allocate()`

### SHA256 Hashing
- Uses OpenSSL's EVP API for cryptographic hashing
- Collision-resistant for cache indexing purposes
- Not designed for authentication or signing

### Concurrent Access
- Cache index uses shared mutexes (read-heavy optimization)
- LRU eviction is protected by exclusive locks
- No deadlock scenarios in current implementation

## Security Updates

Security updates will be released as:
- **Patch versions** (e.g., 1.0.1) for minor issues
- **Minor versions** (e.g., 1.1.0) for important fixes
- **Security advisories** on GitHub for critical issues

Subscribe to:
- [GitHub Security Advisories](https://github.com/ayinedjimi/KVortex/security/advisories)
- [Release notifications](https://github.com/ayinedjimi/KVortex/releases)

## Contact

For security-related questions or concerns:
- **Email:** contact@ayinedjimi-consultants.fr
- **Website:** [ayinedjimi-consultants.fr](https://ayinedjimi-consultants.fr)
- **Professional Services:** Available for security audits and custom deployments

---

**Thank you for helping keep KVortex and its users safe!**
