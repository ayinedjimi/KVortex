// KVortex - High-Performance KV Cache for vLLM
// Copyright (C) 2026 KVortex Contributors
// Based on LMCache (Apache 2.0), Copyright (C) 2024 LMCache Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "config.hpp"
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <chrono>

namespace kvortex {

// ============================================================================
// Logger (Thread-Safe, Simplified)
// ============================================================================

class Logger {
public:
    using LogLevel = KVortexConfig::LogLevel;

    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) {
        std::lock_guard lock(mutex_);
        min_level_ = level;
    }

    void set_output_file(const std::string& path) {
        std::lock_guard lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
        if (!path.empty()) {
            file_.open(path, std::ios::app);
        }
    }

    template <typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
        if (level < min_level_) return;

        auto msg = std::format(fmt, std::forward<Args>(args)...);
        auto timestamp = std::chrono::system_clock::now();

        std::lock_guard lock(mutex_);
        auto& out = file_.is_open() ? file_ : std::cerr;

        out << std::format("[{}] [{}] {}\n",
                          format_timestamp(timestamp),
                          level_str(level),
                          msg);
        out.flush();
    }

    template <typename... Args>
    void trace(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::TRACE, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::WARN, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
    }

private:
    Logger() = default;
    ~Logger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    static constexpr std::string_view level_str(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            default: return "?????";
        }
    }

    static std::string format_timestamp(std::chrono::system_clock::time_point tp) {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            tp.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    LogLevel min_level_{LogLevel::INFO};
    std::mutex mutex_;
    std::ofstream file_;
};

// ============================================================================
// Convenience Macros
// ============================================================================

#define LOG_TRACE(...) ::kvortex::Logger::instance().trace(__VA_ARGS__)
#define LOG_DEBUG(...) ::kvortex::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...)  ::kvortex::Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...)  ::kvortex::Logger::instance().warn(__VA_ARGS__)
#define LOG_ERROR(...) ::kvortex::Logger::instance().error(__VA_ARGS__)

} // namespace kvortex
