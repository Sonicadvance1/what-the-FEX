// SPDX-License-Identifier: MIT
#pragma once
#include <atomic>
#include <cstdint>

// Reimplementation of FEXCore::Profiler struct types.
namespace FEXCore::Profiler {
constexpr uint32_t STATS_VERSION = 2;
enum class AppType : uint8_t {
  LINUX_32,
  LINUX_64,
  WIN_ARM64EC,
  WIN_WOW64,
};

struct ThreadStatsHeader {
  uint8_t Version;
  AppType app_type;
  uint16_t ThreadStatsSize;
  char fex_version[48];
  std::atomic<uint32_t> Head;
  std::atomic<uint32_t> Size;
  uint32_t pad;
};

struct ThreadStats {
  uint32_t Next;
  uint32_t TID;

  // Accumulated time
  uint64_t AccumulatedJITTime;
  uint64_t AccumulatedSignalTime;

  // Accumulated event counts
  uint64_t SIGBUSCount;
  uint64_t SMCCount;
  uint64_t FloatFallbackCount;

  uint64_t AccumulatedCacheMissCount;
  uint64_t AccumulatedCacheReadLockTime;
  uint64_t AccumulatedCacheWriteLockTime;

  uint64_t AccumulatedJITCount;
};
static_assert(sizeof(ThreadStats) % 16 == 0);

static inline const char* GetAppType(FEXCore::Profiler::AppType Type) {
  switch (Type) {
  case FEXCore::Profiler::AppType::LINUX_32: return "Linux32";
  case FEXCore::Profiler::AppType::LINUX_64: return "Linux64";
  case FEXCore::Profiler::AppType::WIN_ARM64EC: return "arm64ec";
  case FEXCore::Profiler::AppType::WIN_WOW64: return "wow64";
  default: break;
  }

  return "Unknown";
}

} // namespace FEXCore::Profiler
