// SPDX-License-Identifier: MIT
#pragma once
#include "ThreadStats.hpp"

#include <cstdint>
#include <ranges>
#include <string_view>
#include <variant>
#include <vector>

namespace Stats {
  enum class AccumulationType {
    InstantAverage,
    Total,
    ExponentialMovingAverage,
  };

  struct AccumulationInfo {
    AccumulationType Type {};
    size_t Offset {};
    std::string_view Name {};
  };

  using AccumulationValueType =  std::variant<uint64_t, double>;
  struct AccumulationValue {
    AccumulationInfo Info {};
    size_t MaxAccumulationAmount {};
    std::vector<uint64_t> Values {};
    AccumulationValueType Accumulation {};
  };

  static inline void Reset(AccumulationValue* Acc) {
    Acc->Values.clear();
    Acc->Accumulation = {};
  }

  template<typename T>
  static inline void inc(AccumulationValueType &Acc, T Value) {
    if (auto val = std::get_if<T>(&Acc)) {
      Acc.emplace<T>(*val + Value);
    } else {
      Acc = Value;
    }
  }

  static inline void Accumulate(AccumulationValue* Acc, FEXCore::Profiler::ThreadStats * const Stats) {
    const auto SampleValue = *reinterpret_cast<const uint64_t*>(reinterpret_cast<uintptr_t>(Stats) + Acc->Info.Offset);

    switch (Acc->Info.Type) {
      case AccumulationType::ExponentialMovingAverage: [[fallthrough]];
      case AccumulationType::InstantAverage: Acc->Values.emplace_back(SampleValue); break;
      case AccumulationType::Total: inc(Acc->Accumulation, SampleValue); break;
    }
  }

  static inline void Average(AccumulationValue* Acc) {
    switch (Acc->Info.Type) {
      case AccumulationType::InstantAverage: {
        uint64_t Result {};
        for (auto Value : Acc->Values) {
          Result += Value;
        }
        double Average = static_cast<double>(Result) / Acc->Values.size();
        Acc->Accumulation = Average;
        break;
      }
      case AccumulationType::Total: /* Nop */ break;
      case AccumulationType::ExponentialMovingAverage: {
        // HdkR: I've never implemented one of these before, so this is best guess.
        constexpr double Alpha = 1.0 / 10.0;
        bool First {true};
        double Result {};
        for (auto Value : std::ranges::reverse_view(Acc->Values)) {
          double FloatValue = Result;
          if (First) {
            Result = FloatValue;
            First = false;
            continue;
          }

          // TODO: Is this correct?
          Result = (FloatValue * Alpha) + (Result * (1.0 - Alpha));
        }

        Acc->Accumulation = Result;
        break;
      }
    }
  }
}
