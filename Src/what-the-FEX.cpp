// SPDX-License-Identifier: MIT
#include "StatsAccumulation.hpp"
#include "WindowStack.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <fcntl.h>
#include <locale.h>
#include <map>
#include <ranges>
#include <sys/poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <thread>
#include <unistd.h>
#include <ncurses.h>
#include <signal.h>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

static const std::array<wchar_t, 10> partial_pips {
  L'\U00002002', // 0%: Empty
  L'\U00002581', // 10%: 1/8 (12.5%)
  L'\U00002581', // 20%: 1/8 (12.5%)
  L'\U00002582', // 30%: 2/8 (25%)
  L'\U00002583', // 40%: 3/8 (37.5%)
  L'\U00002584', // 50%: 4/8 (50%)
  L'\U00002585', // 60%: 5/8 (62.5%)
  L'\U00002586', // 70%: 6/8 (75%)
  L'\U00002587', // 80%: 7/8 (87.5%)
  L'\U00002588', // Full
};

constexpr double NanosecondsInSeconds = 1'000'000'000.0;

struct fex_stats {
  const char *pid_str {};
  int pid {-1};
  int shm_fd {-1};
  bool first_sample = true;
  uint32_t shm_size {};
  uint64_t cycle_counter_frequency {};
  double cycle_counter_frequency_double {};
  size_t hardware_concurrency {};
  size_t page_size {};

  void* shm_base {};
  FEXCore::Profiler::ThreadStatsHeader* head {};
  size_t thread_stats_size_to_copy {};

  WTF::WinStack WinStackMgr;

  struct retained_stats {
    std::chrono::time_point<std::chrono::steady_clock> LastSeen;
    FEXCore::Profiler::ThreadStats PreviousStats {};
    FEXCore::Profiler::ThreadStats Stats {};
  };

  std::chrono::time_point<std::chrono::steady_clock> previous_sample_period;
  std::map<uint32_t, retained_stats> sampled_stats;

  std::wstring empty_pip_data;

  struct max_thread_loads {
    float load_percentage {};
    uint64_t TotalCycles {};
    std::wstring pip_data {};
  };
  std::vector<max_thread_loads> max_thread_loads {};

  std::vector<float> fex_load_histogram;

  struct FEXMemStats final {
    // Total resident
    std::atomic<uint64_t> TotalAnon {~0ULL};

    // JIT Code
    std::atomic<uint64_t> JITCode {~0ULL};
    std::atomic<uint64_t> OpDispatcher {~0ULL};
    std::atomic<uint64_t> Frontend {~0ULL};
    std::atomic<uint64_t> CPUBackend {~0ULL};
    std::atomic<uint64_t> Lookup {~0ULL};
    std::atomic<uint64_t> LookupL1 {~0ULL};
    std::atomic<uint64_t> ThreadStates {~0ULL};
    std::atomic<uint64_t> BlockLinks {~0ULL};
    std::atomic<uint64_t> Misc {~0ULL};
    std::atomic<uint64_t> JEMalloc {~0ULL};
    std::atomic<uint64_t> Unaccounted {~0ULL};

    struct LargestAnonType {
      uint64_t Begin, End;
      uint64_t Size;
    };
    LargestAnonType LargestAnon;
  };
  FEXMemStats MemStats;

  std::atomic<bool> ShuttingDown {};

  int pidfd_watch {-1};

  fex_stats()
    : fex_load_histogram(200, 0.0f) {}
};

auto SamplePeriod = std::chrono::milliseconds(1000);
fex_stats g_stats {};

#ifndef __x86_64__
uint64_t get_cycle_counter_frequency() {
  uint64_t result;
  __asm("mrs %[Res], CNTFRQ_EL0;\n" : [Res] "=r"(result));

  return result;
}
static void store_memory_barrier() {
  asm volatile("dmb ishst" ::: "memory");
}
#else
static uint64_t get_cycle_counter_frequency() {
  return 1;
}
static void store_memory_barrier() {}
#endif

static FEXCore::Profiler::ThreadStats* StatFromOffset(void* Base, uint32_t Offset) {
  return reinterpret_cast<FEXCore::Profiler::ThreadStats*>(reinterpret_cast<uint64_t>(Base) + Offset);
}

static void exit_screen(const char* format = nullptr, ...) {
  refresh();
  endwin();

  if (format != nullptr) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
  }
  _exit(0);
}

static void handle_signal(int signum, siginfo_t* info, void* context) {
  exit_screen();
}

static void setup_signal_handler() {
  struct sigaction sa;
  sa.sa_sigaction = handle_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGQUIT, &sa, NULL);
}

static void check_shm_update_necessary() {
  auto new_shm_size = g_stats.head->Size.load(std::memory_order_relaxed);
  if (g_stats.shm_size != new_shm_size) {
    // Remap!
    munmap(g_stats.shm_base, g_stats.shm_size);
    g_stats.shm_size = new_shm_size;
    g_stats.shm_base = mmap(nullptr, new_shm_size, PROT_READ, MAP_SHARED, g_stats.shm_fd, 0);

    // Update head pointer as well.
    g_stats.head = reinterpret_cast<FEXCore::Profiler::ThreadStatsHeader*>(g_stats.shm_base);
  }
}

static uint64_t ConvertToBytes(std::string_view Size, std::string_view Granule) {
  uint64_t SizeBytes {};
  SizeBytes = strtoull(Size.data(), nullptr, 10);

  // Granule should only be in kB.
  if (Granule == "kB") {
    SizeBytes *= 1024U;
  }
  else {
    exit_screen("Unknown size modifier: %s\n", std::string(Granule).c_str());
  }

  return SizeBytes;
}

static std::string ConvertMemToHuman(uint64_t MemBytes) {
  const char *Granule;
  if (MemBytes >= (1024 * 1024)) {
    MemBytes /= 1024 * 1024;
    Granule = "MiB";
  } else if (MemBytes >= 1024) {
    MemBytes /= 1024;
    Granule = "KiB";
  }
  return std::format("{} {}", MemBytes, Granule);
}

static void ResidentFEXAnonSampling() {
  const auto fex_pid_smaps = std::format("/proc/{}/smaps", g_stats.pid);

  const int smap_fd = open(fex_pid_smaps.c_str(), O_RDONLY);

  if (smap_fd == -1) return;

  std::string File{};
  while (!g_stats.ShuttingDown.load()) {

    // Read the full file again.
    File.clear();
    lseek(smap_fd, 0, SEEK_SET);
    char TempBuffer[4096];
    ssize_t ReadSize {};
    while ((ReadSize = read(smap_fd, TempBuffer, 4096)) > 0) {
      File.append(TempBuffer, ReadSize);
    }

    if (ReadSize == -1) {
      // Error.
      goto exit;
    }

    // Parse the file by line.
    uint64_t TotalResident {};
    uint64_t TotalJITResident {};
    uint64_t TotalOpDispatcherResident {};
    uint64_t TotalFrontendResident {};
    uint64_t TotalCPUBackendResident {};
    uint64_t TotalLookupResident {};
    uint64_t TotalLookupL1Resident {};
    uint64_t TotalThreadStateResident {};
    uint64_t TotalBlockLinksResident {};
    uint64_t TotalMiscResident{};
    uint64_t TotalJEMallocResident{};
    uint64_t TotalUnaccounted{};
    fex_stats::FEXMemStats::LargestAnonType LargestRSSAnon {};

    uint64_t Begin {}, End {};
    uint64_t *ActiveSubRegion {};

    std::istringstream ss(File);
    std::string Line;
    while (std::getline(ss, Line)) {
      // `359519000-359918000 ---p 00000000 00:00 0                                [anon:FEXMem]`
      if (Line.find("FEXMem") != Line.npos) {
        sscanf(Line.c_str(), "%lx-%lx", &Begin, &End);

        if (Line.find("FEXMemJIT") != Line.npos) {
          ActiveSubRegion = &TotalJITResident;
        } else if (Line.find("FEXMem_OpDispatcher") != Line.npos) {
          ActiveSubRegion = &TotalOpDispatcherResident;
        } else if (Line.find("FEXMem_Frontend") != Line.npos) {
          ActiveSubRegion = &TotalFrontendResident;
        } else if (Line.find("FEXMem_CPUBackend") != Line.npos) {
          ActiveSubRegion = &TotalCPUBackendResident;
        } else if (Line.find("FEXMem_Lookup_L1") != Line.npos) {
          ActiveSubRegion = &TotalLookupL1Resident;
        } else if (Line.find("FEXMem_Lookup") != Line.npos) {
          ActiveSubRegion = &TotalLookupResident;
        } else if (Line.find("FEXMem_ThreadState") != Line.npos) {
          ActiveSubRegion = &TotalThreadStateResident;
        } else if (Line.find("FEXMem_BlockLinks") != Line.npos) {
          ActiveSubRegion = &TotalBlockLinksResident;
        } else if (Line.find("FEXMem_Misc") != Line.npos) {
          ActiveSubRegion = &TotalMiscResident;
        } else {
          // Fully anonymous.
          ActiveSubRegion = &TotalUnaccounted;
        }

        continue;
      }

      if (Line.find("JEMalloc") != Line.npos || Line.find("FEXAllocator") != Line.npos) {
        ActiveSubRegion = &TotalJEMallocResident;
        sscanf(Line.c_str(), "%lx-%lx", &Begin, &End);
        continue;
      }

      if (Line.find("VmFlags") != Line.npos) {
        ActiveSubRegion = nullptr;
        continue;
      }

      if (ActiveSubRegion && Line.find("Rss") != Line.npos) {
        // Parse the residency for this mapped region and add it.
        // ex: `Rss:                 560 kB`
        auto GranuleIter = Line.find_last_of(' ') + 1;
        auto SizeIter = Line.find_last_of(' ', GranuleIter - 2) + 1;

        std::string_view GranuleView = std::string_view(&Line.at(GranuleIter));
        std::string_view SizeView = std::string_view(&Line.at(SizeIter), GranuleIter - SizeIter - 1);
        uint64_t ResidentInBytes = ConvertToBytes(SizeView, GranuleView);
        TotalResident += ResidentInBytes;
        *ActiveSubRegion += ResidentInBytes;

        if (ActiveSubRegion == &TotalJEMallocResident) {
          if (LargestRSSAnon.Size < ResidentInBytes) {
            LargestRSSAnon = {
              .Begin = Begin,
              .End = End,
              .Size = ResidentInBytes,
            };
          }
        }
        continue;
      }
    }

    if (TotalResident) {
      g_stats.MemStats.LargestAnon = LargestRSSAnon;

      g_stats.MemStats.TotalAnon.store(TotalResident);
      g_stats.MemStats.JITCode.store(TotalJITResident);
      g_stats.MemStats.OpDispatcher.store(TotalOpDispatcherResident);
      g_stats.MemStats.Frontend.store(TotalFrontendResident);
      g_stats.MemStats.CPUBackend.store(TotalCPUBackendResident);
      g_stats.MemStats.Lookup.store(TotalLookupResident);
      g_stats.MemStats.LookupL1.store(TotalLookupL1Resident);
      g_stats.MemStats.ThreadStates.store(TotalThreadStateResident);
      g_stats.MemStats.BlockLinks.store(TotalBlockLinksResident);
      g_stats.MemStats.Misc.store(TotalMiscResident);
      g_stats.MemStats.JEMalloc.store(TotalJEMallocResident);
      g_stats.MemStats.Unaccounted.store(TotalUnaccounted);
    }
    std::this_thread::sleep_for(SamplePeriod);
  }

exit:
  close(smap_fd);
}

static uint64_t CyclesToMilliseconds(uint64_t Cycles) {
  const double Cycles_f = Cycles;
  const double CyclesPerMillisecond = g_stats.cycle_counter_frequency_double / 1000.0;
  return Cycles_f / CyclesPerMillisecond;
}

static std::string CustomPrintInteger(uint64_t Integer) {
  // Maximum integer that can fit in to uint64_t, plus commas, plus null
  char buf[27] {};
  auto result = std::to_chars(buf, buf + sizeof(buf), Integer, 10);

  const auto size = result.ptr - buf;
  for (ssize_t i = size - 3; i > 0; i -= 3) {
    const auto remaining_size = size - i;
    memmove(&buf[i + 1], &buf[i], remaining_size + 1);
    buf[i] = ',';
  }

  return std::string(buf);
}

static void SampleStats(std::chrono::steady_clock::time_point Now) {
  auto AtomicCopyStats = [](FEXCore::Profiler::ThreadStats* Dest, FEXCore::Profiler::ThreadStats* Src, size_t Size) {
    // Take advantage of 16-byte alignment and single-copy atomicity of ARMv8.4.
#if defined(__x86_64__) || defined(__i386__)
    using copy_type = __m128;
#else
    using copy_type = __uint128_t;
#endif
    const auto elements_to_copy = g_stats.thread_stats_size_to_copy / sizeof(copy_type);
    auto d_i = reinterpret_cast<copy_type*>(Dest);
    auto s_i = reinterpret_cast<const copy_type*>(Src);
    for (size_t i = 0; i < elements_to_copy; ++i) {
      d_i[i] = s_i[i];
    }
  };

  uint32_t HeaderOffset = g_stats.head->Head;
  while (HeaderOffset != 0) {
    if (HeaderOffset >= g_stats.shm_size) {
      break;
    }
    FEXCore::Profiler::ThreadStats* Stat = StatFromOffset(g_stats.shm_base, HeaderOffset);

    auto it = &g_stats.sampled_stats[Stat->TID];
    AtomicCopyStats(&it->Stats, Stat, g_stats.thread_stats_size_to_copy);
    it->LastSeen = Now;

    HeaderOffset = Stat->Next;
  }
}

static int selected = 0;
bool Collapsed[3] {};

void HandleSelectMove(int c) {
  // TODO: Find a better way to update sample period.
  if (false) {
    if (c == KEY_UP) {
      if (SamplePeriod > std::chrono::milliseconds(100)) {
        SamplePeriod = std::min(SamplePeriod + std::chrono::milliseconds(100), std::chrono::milliseconds(1000));
      }
      else {
        SamplePeriod = std::max(SamplePeriod + std::chrono::milliseconds(10), std::chrono::milliseconds(10));
      }
    } else if (c == KEY_DOWN) {
      if (SamplePeriod > std::chrono::milliseconds(100)) {
        SamplePeriod = std::max(SamplePeriod - std::chrono::milliseconds(100), std::chrono::milliseconds(100));
      }
      else {
        SamplePeriod = std::max(SamplePeriod - std::chrono::milliseconds(10), std::chrono::milliseconds(10));
      }
    }
  }

  if (c == KEY_UP) {
    if (selected > 0) {
      --selected;
    }
  }
  else if (c == KEY_DOWN) {
    if (selected < 2) {
      ++selected;
    }
  }
  else if (c == KEY_RIGHT) {
    Collapsed[selected] ^= true;
  }
}

wchar_t Selected[2] = {
  L'☐',
  L'*',
};

wchar_t CollapsedItem[2] = {
  L'▼',
  L'►',
};

void HandleHistogram(WINDOW *win, void* user_data) {
  const auto win_height = getmaxy(win);
  const auto win_width = getmaxx(win);

  const auto WIN_INDEX = 2;
  const auto WIN_NAME = "Total JIT usage";
  const bool WinCollapsed = Collapsed[WIN_INDEX];
  if (!WinCollapsed) {
    g_stats.WinStackMgr.RequestNewHeight(WIN_INDEX, 12);
  } else if (WinCollapsed) {
    g_stats.WinStackMgr.RequestNewHeight(WIN_INDEX, 1);
  }

  if (!WinCollapsed) {
    const auto HistogramHeight = win_height - 2;
    size_t HistogramWidth = win_width - 2;
    HistogramWidth = std::min(HistogramWidth, g_stats.fex_load_histogram.size());

    size_t j = 0;

    for (auto& HistogramResult : std::ranges::reverse_view {g_stats.fex_load_histogram}) {
      for (size_t i = 0; i < HistogramHeight; ++i) {
        int attr = 0;
        if (HistogramResult >= 75.0) {
          attr = 1;
        } else if (HistogramResult >= 50.0) {
          attr = 2;
        }
        if (attr) {
          wattron(win, COLOR_PAIR(attr));
        }

        double rounded_down = std::floor(HistogramResult / 10.0) * 10.0;
        size_t tens_digit = rounded_down / 10.0;
        size_t digit_percent = std::floor(HistogramResult - rounded_down);

        size_t pip = 0;
        if (tens_digit > i) {
          pip = partial_pips.size() - 1;
        } else if (tens_digit == i) {
          pip = digit_percent;
        }

        mvwprintw(win, HistogramHeight - i, win_width - j - 2, "%lc", partial_pips[pip]);
        if (attr) {
          wattroff(win, COLOR_PAIR(attr));
        }
      }
      ++j;
    }
  }

  box(win, 0, 0);
  bool IsSelected = selected == WIN_INDEX;
  mvwprintw(win, 0, 0, "%lc %lc %s", Selected[IsSelected], CollapsedItem[WinCollapsed ? 1 : 0], WIN_NAME);
}

void HandleMemstats(WINDOW *win, void* user_data) {
  const auto WIN_INDEX = 1;
  const auto WIN_NAME = "FEX Memory Usage";
  const bool WinCollapsed = Collapsed[WIN_INDEX];
  if (!WinCollapsed) {
    g_stats.WinStackMgr.RequestNewHeight(WIN_INDEX, 15);
  } else if (WinCollapsed) {
    g_stats.WinStackMgr.RequestNewHeight(WIN_INDEX, 1);
  }

  if (!WinCollapsed) {
    const uint64_t MemBytes = g_stats.MemStats.TotalAnon.load();
    const uint64_t MemBytesJIT = g_stats.MemStats.JITCode.load();
    const uint64_t MemBytesOpDispatcher = g_stats.MemStats.OpDispatcher.load();
    const uint64_t MemBytesFrontend = g_stats.MemStats.Frontend.load();
    const uint64_t MemBytesCPUBackend = g_stats.MemStats.CPUBackend.load();
    const uint64_t MemBytesLookup = g_stats.MemStats.Lookup.load();
    const uint64_t MemBytesLookupL1 = g_stats.MemStats.LookupL1.load();
    const uint64_t MemBytesThreadStates = g_stats.MemStats.ThreadStates.load();
    const uint64_t MemBytesBlockLinks = g_stats.MemStats.BlockLinks.load();
    const uint64_t MemBytesMisc = g_stats.MemStats.Misc.load();
    const uint64_t MemBytesJEMalloc = g_stats.MemStats.JEMalloc.load();
    const uint64_t MemBytesUnaccounted = g_stats.MemStats.Unaccounted.load();

    constexpr static size_t TotalMemLines = 11;

    if (MemBytes == ~0ULL) {
      mvwprintw(win, 1, 1, "Total FEX Anon memory resident: Couldn't detect\n");
    } else {
      std::string SizeHuman = ConvertMemToHuman(MemBytes);
      std::string SizeHumanJIT = ConvertMemToHuman(MemBytesJIT);
      std::string SizeHumanOpDispatcher = ConvertMemToHuman(MemBytesOpDispatcher);
      std::string SizeHumanFrontend = ConvertMemToHuman(MemBytesFrontend);
      std::string SizeHumanCPUBackend = ConvertMemToHuman(MemBytesCPUBackend);
      std::string SizeHumanLookup = ConvertMemToHuman(MemBytesLookup);
      std::string SizeHumanLookupL1 = ConvertMemToHuman(MemBytesLookupL1);
      std::string SizeHumanThreadStates = ConvertMemToHuman(MemBytesThreadStates);
      std::string SizeHumanBlockLinks = ConvertMemToHuman(MemBytesBlockLinks);
      std::string SizeHumanMisc = ConvertMemToHuman(MemBytesMisc);
      std::string SizeHumanJEMalloc = ConvertMemToHuman(MemBytesJEMalloc);
      std::string SizeHumanUnaccounted = ConvertMemToHuman(MemBytesUnaccounted);
      std::string SizeHumanLargestUnaccounted = ConvertMemToHuman(g_stats.MemStats.LargestAnon.Size);

      mvwprintw(win, 1, 1,  "Total FEX Anon memory resident: %s\n", SizeHuman.c_str());
      mvwprintw(win, 2, 1,  "    JIT resident:             %s\n", SizeHumanJIT.c_str());
      mvwprintw(win, 3, 1,  "    OpDispatcher resident:    %s\n", SizeHumanOpDispatcher.c_str());
      mvwprintw(win, 4, 1,  "    Frontend resident:        %s\n", SizeHumanFrontend.c_str());
      mvwprintw(win, 5, 1,  "    CPUBackend resident:      %s\n", SizeHumanCPUBackend.c_str());
      mvwprintw(win, 6, 1,  "    Lookup cache resident:    %s\n", SizeHumanLookup.c_str());
      mvwprintw(win, 7, 1,  "    Lookup L1 cache resident: %s\n", SizeHumanLookupL1.c_str());
      mvwprintw(win, 8, 1,  "    ThreadStates resident:    %s\n", SizeHumanThreadStates.c_str());
      mvwprintw(win, 9, 1,  "    BlockLinks resident:      %s\n", SizeHumanBlockLinks.c_str());
      mvwprintw(win, 10, 1,  "          Misc resident:      %s\n", SizeHumanMisc.c_str());
      mvwprintw(win, 11, 1, "    JEMalloc resident:        %s\n", SizeHumanJEMalloc.c_str());
      mvwprintw(win, 12, 1, "    Unaccounted resident:     %s\n", SizeHumanUnaccounted.c_str());
      mvwprintw(win, 13, 1, "                 Largest:     %s [0x%lx, 0x%lx) - p (void*) memset(0x%lx, 0xFF, %ld)\n",
          SizeHumanLargestUnaccounted.c_str(), g_stats.MemStats.LargestAnon.Begin, g_stats.MemStats.LargestAnon.End, g_stats.MemStats.LargestAnon.Begin, g_stats.MemStats.LargestAnon.End - g_stats.MemStats.LargestAnon.Begin);
    }
  }

  box(win, 0, 0);
  bool IsSelected = selected == WIN_INDEX;
  mvwprintw(win, 0, 0, "%lc %lc %s", Selected[IsSelected], CollapsedItem[WinCollapsed ? 1 : 0], WIN_NAME);
}

struct JITStatsUserData {
  FEXCore::Profiler::ThreadStats TotalThisPeriod;
  std::vector<uint64_t> hottest_threads;
  std::chrono::nanoseconds sample_period;
  size_t threads_sampled {};
  uint64_t total_jit_time {};
  uint64_t TotalJITInvocations {};
  double fex_load {};
  double Scale = 1000.0;
  const char* ScaleStr = "ms/second";
};

void HandleJITstats(WINDOW *win, void* user_data) {
  const auto win_height = getmaxy(win);
  const auto win_width = getmaxx(win);

  const auto WIN_INDEX = 0;
  const auto WIN_NAME = "FEX JIT Stats";
  const bool WinCollapsed = Collapsed[WIN_INDEX];
  if (!WinCollapsed) {
    g_stats.WinStackMgr.RequestNewHeight(WIN_INDEX, 26);
  } else if (WinCollapsed) {
    g_stats.WinStackMgr.RequestNewHeight(WIN_INDEX, 1);
  }

  if (!WinCollapsed) {
    const auto JITData = reinterpret_cast<const JITStatsUserData*>(user_data);
    const auto& TotalThisPeriod = JITData->TotalThisPeriod;
    const auto& hottest_threads = JITData->hottest_threads;
    const auto& sample_period = JITData->sample_period;
    const auto& threads_sampled = JITData->threads_sampled;
    const auto& total_jit_time = JITData->total_jit_time;
    const auto& TotalJITInvocations = JITData->TotalJITInvocations;
    const auto& Scale = JITData->Scale;
    const auto& ScaleStr = JITData->ScaleStr;

    const auto JITSeconds = (double)(TotalThisPeriod.AccumulatedJITTime) / g_stats.cycle_counter_frequency_double;
    const auto SignalTime = (double)(TotalThisPeriod.AccumulatedSignalTime) / g_stats.cycle_counter_frequency_double;

    const auto SIGBUSCount = TotalThisPeriod.SIGBUSCount;
    const auto SMCCount = TotalThisPeriod.SMCCount;
    const auto FloatFallbackCount = TotalThisPeriod.FloatFallbackCount;
    const auto AccumulatedCacheMissCount = TotalThisPeriod.AccumulatedCacheMissCount;
    const auto AccumulatedCacheReadLockTime = (double)(TotalThisPeriod.AccumulatedCacheReadLockTime) / g_stats.cycle_counter_frequency_double;
    const auto AccumulatedCacheWriteLockTime = (double)(TotalThisPeriod.AccumulatedCacheWriteLockTime) / g_stats.cycle_counter_frequency_double;
    const auto AccumulatedJITCount = TotalThisPeriod.AccumulatedJITCount;

    const auto MaxActiveThreads = std::min<size_t>(g_stats.sampled_stats.size(), std::min<size_t>(g_stats.hardware_concurrency, 32));

    mvwprintw(win, 1, 1, "Top %ld threads executing (%ld total)\n", g_stats.max_thread_loads.size(), threads_sampled);

    size_t max_pips = std::min(win_width, 50) - 2;
    double percentage_per_pip = 100.0 / (double)max_pips;

    g_stats.empty_pip_data.resize(max_pips);
    std::fill(g_stats.empty_pip_data.begin(), g_stats.empty_pip_data.begin() + max_pips, partial_pips.front());
    size_t i = 0;
    for (auto &thread_loads : std::ranges::reverse_view {g_stats.max_thread_loads}) {
      double thread_load = std::min(thread_loads.load_percentage, 100.0f);
      thread_loads.pip_data.resize(max_pips);
      double rounded_down = std::floor(thread_load / 10.0) * 10.0;
      size_t full_pips = rounded_down / percentage_per_pip;
      size_t digit_percent = thread_load - rounded_down;
      wmemset(thread_loads.pip_data.data(), partial_pips.front(), thread_loads.pip_data.size());
      wmemset(thread_loads.pip_data.data(), partial_pips.back(), full_pips);
      wmemset(thread_loads.pip_data.data() + full_pips, partial_pips[digit_percent], 1);

      const auto y_offset = 2 + i;
      mvwprintw(win, y_offset, 1, "[%ls]: %.02f%% (%zd ms/S, %zd cycles)\n", g_stats.empty_pip_data.data(), thread_load, CyclesToMilliseconds(thread_loads.TotalCycles), thread_loads.TotalCycles);
      int attr = 0;
      if (thread_load >= 75.0) {
        attr = 1;
      } else if (thread_load >= 50.0) {
        attr = 2;
      }
      if (attr) {
        attron(COLOR_PAIR(attr));
      }
      mvwprintw(win, y_offset, 1, "[%ls]", thread_loads.pip_data.data());
      if (attr) {
        attroff(COLOR_PAIR(attr));
      }
      ++i;
    }

    const double SamplePeriodNanoseconds = sample_period.count();

    const double SIGBUS_l = SIGBUSCount;
    const double SIGBUS_Per_Second = SIGBUS_l * (SamplePeriodNanoseconds / NanosecondsInSeconds);

    const double AccumulatedCacheMissCount_l = AccumulatedCacheMissCount;
    const double AccumulatedCacheMissCount_Per_Second = AccumulatedCacheMissCount_l * (SamplePeriodNanoseconds / NanosecondsInSeconds);

    const double AccumulatedJITCount_l = AccumulatedJITCount;
    const double AccumulatedJITCount_Per_Second = AccumulatedJITCount_l * (SamplePeriodNanoseconds / NanosecondsInSeconds);

    mvwprintw(win, win_height - 12, 1, "Total (%zd millisecond sample period):\n", std::chrono::duration_cast<std::chrono::milliseconds>(SamplePeriod).count());
    mvwprintw(win, win_height - 11, 1, "       JIT Time: %f %s (%.2f percent)\n", JITSeconds * Scale, ScaleStr, JITSeconds / (double)MaxActiveThreads * 100.0);
    mvwprintw(win, win_height - 10, 1, "    Signal Time: %f %s (%.2f percent)\n", SignalTime * Scale, ScaleStr, SignalTime / (double)MaxActiveThreads * 100.0);
    mvwprintw(win, win_height - 9, 1,  "     SIGBUS Cnt: %ld (%lf per second)\n", SIGBUSCount, SIGBUS_Per_Second);
    mvwprintw(win, win_height - 8, 1,  "        SMC Cnt: %ld\n", SMCCount);
    mvwprintw(win, win_height - 7, 1,  "  Softfloat Cnt: %s\n", CustomPrintInteger(FloatFallbackCount).c_str());
    mvwprintw(win, win_height - 6, 1,  "  CacheMiss Cnt: %ld (%lf per second) (%s total JIT invocations)\n", AccumulatedCacheMissCount, AccumulatedCacheMissCount_Per_Second, CustomPrintInteger(TotalJITInvocations).c_str());
    mvwprintw(win, win_height - 5, 1,  "    $RDLck Time: %f %s (%.2f percent)\n", AccumulatedCacheReadLockTime * Scale, ScaleStr, AccumulatedCacheReadLockTime / (double)MaxActiveThreads * 100.0);
    mvwprintw(win, win_height - 4, 1,  "    $WRLck Time: %f %s (%.2f percent)\n", AccumulatedCacheWriteLockTime * Scale, ScaleStr, AccumulatedCacheWriteLockTime / (double)MaxActiveThreads * 100.0);
    mvwprintw(win, win_height - 3, 1,  "        JIT Cnt: %ld (%lf percent)\n", AccumulatedJITCount, AccumulatedJITCount_Per_Second);
    mvwprintw(win, win_height - 2, 1,  "FEX JIT Load:    %f (cycles: %ld)\n", JITData->fex_load, total_jit_time);

    // <Box> + <Lines of text> + <Thread stats> + <Top %d threads executing text>
    const int Height = 2 + 11 + MaxActiveThreads + 1;
    if (Height != win_height) {
      g_stats.WinStackMgr.RequestNewHeight(0, Height);
    }
  }

  box(win, 0, 0);
  bool IsSelected = selected == WIN_INDEX;
  mvwprintw(win, 0, 0, "%lc %lc %s", Selected[IsSelected], CollapsedItem[WinCollapsed ? 1 : 0], WIN_NAME);

  char buffer[64];
  int cx = snprintf(buffer, sizeof(buffer), "PID: %d", g_stats.pid);

  mvwprintw(win, 0, win_width - cx - 1, "%s", buffer);
}

void AppendJITstatsSubWin(WTF::WinStack *WinStackMgr, WINDOW *main, JITStatsUserData *JITData) {
  int lines = 26;
  int cols = COLS;
  int y = 0;
  int x = 0;
  auto win = subwin(main, lines, cols, y, x);
  WinStackMgr->AddToStack(HandleJITstats, win, JITData, WTF::WinStack::Properties {
    .Height = lines,
  });
}

void AppendMemstatsSubWin(WTF::WinStack *WinStackMgr, WINDOW *main) {
  int lines = 15;
  int cols = COLS;
  int y = 0;
  int x = 0;
  auto win = subwin(main, lines, cols, y, x);
  WinStackMgr->AddToStack(HandleMemstats, win, nullptr, WTF::WinStack::Properties {
    .Height = lines,
  });
}

void AppendGraphSubWin(WTF::WinStack *WinStackMgr, WINDOW *main) {
  int lines = 12;
  int cols = COLS;
  int y = 0;
  int x = 0;
  auto win = subwin(main, lines, cols, y, x);
  WinStackMgr->AddToStack(HandleHistogram, win, nullptr, WTF::WinStack::Properties {
    .Height = lines,
  });
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s [options] <pid>\n", argv[0]);
    return 0;
  }
  g_stats.pid_str = argv[argc - 1];
  g_stats.pid = strtol(g_stats.pid_str, nullptr, 10);

  setup_signal_handler();

  const auto fex_shm = std::format("fex-{}-stats", g_stats.pid_str);
  g_stats.shm_fd = shm_open(fex_shm.c_str(), O_RDONLY, 0);
  if (g_stats.shm_fd == -1) {
    printf("%s doesn't seem to exist\n", fex_shm.c_str());
    return 1;
  }

  struct stat buf {};
  if (fstat(g_stats.shm_fd, &buf) == -1) {
    printf("Couldn't stat\n");
    return 1;
  }

  if (buf.st_size < sizeof(uint64_t) * 4) {
    printf("Buffer was too small: %ld\n", buf.st_size);
    return 1;
  }

  g_stats.pidfd_watch = ::syscall(SYS_pidfd_open, g_stats.pid, 0);
  setlocale(LC_ALL, "");
  auto window = initscr();
  nodelay(window, true);
  keypad(window, true);
  start_color();
  init_pair(1, COLOR_RED, COLOR_BLACK);
  init_pair(2, COLOR_YELLOW, COLOR_BLACK);

  g_stats.shm_size = buf.st_size;
  g_stats.shm_base = mmap(nullptr, g_stats.shm_size, PROT_READ, MAP_SHARED, g_stats.shm_fd, 0);
  g_stats.head = reinterpret_cast<FEXCore::Profiler::ThreadStatsHeader*>(g_stats.shm_base);

  std::string fex_version {g_stats.head->fex_version, strnlen(g_stats.head->fex_version, sizeof(g_stats.head->fex_version))};

  store_memory_barrier();
  printw("Header for PID %d:\n", g_stats.pid);
  printw("  Version: 0x%x\n", g_stats.head->Version);
  printw("  Type: %s\n", GetAppType(g_stats.head->app_type));
  printw("  Fex: %s\n", fex_version.c_str());
  printw("  Head: 0x%x\n", g_stats.head->Head.load(std::memory_order_relaxed));
  printw("  Size: 0x%x\n", g_stats.head->Size.load(std::memory_order_relaxed));

  if (g_stats.head->Version != FEXCore::Profiler::STATS_VERSION) {
    exit_screen("Unhandled FEX stats version\n");
  }

  g_stats.thread_stats_size_to_copy = sizeof(FEXCore::Profiler::ThreadStats);
  if (g_stats.head->ThreadStatsSize) {
    g_stats.thread_stats_size_to_copy = std::min<size_t>(g_stats.head->ThreadStatsSize, g_stats.thread_stats_size_to_copy);
  }

  g_stats.cycle_counter_frequency = get_cycle_counter_frequency();
  g_stats.cycle_counter_frequency_double = (double)g_stats.cycle_counter_frequency;

  g_stats.hardware_concurrency = std::thread::hardware_concurrency();
  g_stats.max_thread_loads.reserve(g_stats.hardware_concurrency);

  bool FirstLoop = true;
  std::thread ResidentAnonThread {ResidentFEXAnonSampling};

  const char *ExitString {};

  JITStatsUserData JITData {};

  AppendJITstatsSubWin(&g_stats.WinStackMgr, window, &JITData);
  AppendMemstatsSubWin(&g_stats.WinStackMgr, window);
  AppendGraphSubWin(&g_stats.WinStackMgr, window);

  while (true) {
    if (g_stats.pidfd_watch != -1) {
      pollfd fd {
        .fd = g_stats.pidfd_watch,
        .events = POLLIN | POLLHUP,
        .revents = 0,
      };
      int Res = poll(&fd, 1, 0);
      if (Res == 1) {
        if (fd.revents & POLLHUP) {
          ExitString = "FEX process exited\n";
          goto exit;
        }
      }
    }

    auto Now = std::chrono::steady_clock::now();
    auto CurrentSamples = (Now - g_stats.previous_sample_period);
    if (CurrentSamples >= SamplePeriod) {
      // Say our remaining sample is max wait.
      CurrentSamples = std::chrono::milliseconds(10);

      JITData.total_jit_time = {};
      JITData.threads_sampled = {};
      JITData.hottest_threads = {};
      JITData.TotalJITInvocations = {};
      JITData.TotalThisPeriod = {};

      // The writer side doesn't use atomics. Use a memory barrier to ensure writes are visible.
      store_memory_barrier();

      check_shm_update_necessary();

      // Sample the stats from the process. Try and be as quick as possible.
      SampleStats(Now);

#define accumulate(dest, name) dest += Stat->name - PreviousStats->name
      for (auto it = g_stats.sampled_stats.begin(); it != g_stats.sampled_stats.end();) {
        ++JITData.threads_sampled;
        auto PreviousStats = &it->second.PreviousStats;
        auto Stat = &it->second.Stats;
        uint64_t total_time {};

        accumulate(total_time, AccumulatedJITTime);
        accumulate(total_time, AccumulatedSignalTime);
        JITData.total_jit_time += total_time;

        accumulate(JITData.TotalThisPeriod.AccumulatedJITTime, AccumulatedJITTime);
        accumulate(JITData.TotalThisPeriod.AccumulatedSignalTime, AccumulatedSignalTime);

        accumulate(JITData.TotalThisPeriod.SIGBUSCount, SIGBUSCount);
        accumulate(JITData.TotalThisPeriod.SMCCount, SMCCount);
        accumulate(JITData.TotalThisPeriod.FloatFallbackCount, FloatFallbackCount);
        accumulate(JITData.TotalThisPeriod.AccumulatedCacheMissCount, AccumulatedCacheMissCount);
        accumulate(JITData.TotalThisPeriod.AccumulatedCacheReadLockTime, AccumulatedCacheReadLockTime);
        accumulate(JITData.TotalThisPeriod.AccumulatedCacheWriteLockTime, AccumulatedCacheWriteLockTime);
        accumulate(JITData.TotalThisPeriod.AccumulatedJITCount, AccumulatedJITCount);
        JITData.TotalJITInvocations += Stat->AccumulatedJITCount;

        memcpy(PreviousStats, Stat, g_stats.thread_stats_size_to_copy);

        if ((Now - it->second.LastSeen) >= std::chrono::seconds(10)) {
          it = g_stats.sampled_stats.erase(it);
          continue;
        }

        JITData.hottest_threads.emplace_back(total_time);

        ++it;
      }

      std::sort(JITData.hottest_threads.begin(), JITData.hottest_threads.end(), std::greater<uint64_t>());

      // Calculate loads based on the sample period that occurred.
      // FEX-Emu only counts cycles for the amount of time, so we need to calculate load based on the number of cycles that the sample period has.
      JITData.sample_period = Now - g_stats.previous_sample_period;

      const double SamplePeriodNanoseconds = JITData.sample_period.count();
      const double MaximumCyclesInSecond = g_stats.cycle_counter_frequency_double;
      const double MaximumCyclesInSamplePeriod = MaximumCyclesInSecond * (SamplePeriodNanoseconds / NanosecondsInSeconds);
      const double MaximumCoresThreadsPossible = std::min(g_stats.hardware_concurrency, JITData.threads_sampled);
      JITData.fex_load = ((double)JITData.total_jit_time / (MaximumCyclesInSamplePeriod * MaximumCoresThreadsPossible)) * 100.0;

      size_t minimum_hot_threads = std::min(g_stats.hardware_concurrency, JITData.hottest_threads.size());
      // For the top thread-loads, we are only ever showing up to how many hardware threads are available.
      g_stats.max_thread_loads.resize(minimum_hot_threads);
      for (size_t i = 0; i < minimum_hot_threads; ++i) {
        g_stats.max_thread_loads[i].load_percentage = ((double)JITData.hottest_threads[i] / MaximumCyclesInSamplePeriod) * 100.0;
        g_stats.max_thread_loads[i].TotalCycles = JITData.hottest_threads[i];
      }

      g_stats.fex_load_histogram.erase(g_stats.fex_load_histogram.begin());
      g_stats.fex_load_histogram.push_back(JITData.fex_load);

      FirstLoop = false;

      g_stats.previous_sample_period = Now;
    }

    touchwin(window);
    g_stats.WinStackMgr.UpdateWindowDimensions();
    g_stats.WinStackMgr.RunStack();

    int c = wgetch(window);
    HandleSelectMove(c);
    refresh();

    // We want to sleep for at most 10ms. But because our sample period and our loop period is a difference cadence...
    std::this_thread::sleep_for(std::chrono::milliseconds(std::min<size_t>(CurrentSamples.count(), 10)));
  }

exit:
  g_stats.ShuttingDown = true;
  close(g_stats.shm_fd);
  close(g_stats.pidfd_watch);
  ResidentAnonThread.join();
  exit_screen(ExitString);
  return 0;
}
