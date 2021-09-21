#pragma once

// Must come before spdlog.h
#if defined(DEBUG_BUILD)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif

#include <spdlog/spdlog.h>

#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)

namespace Utils
{
inline auto InitializeLogger()
{
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%H:%M:%S] [thread %t] [%l] [%!] %v");
}
}