#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
class PassManager;
}  // namespace mlir

namespace matcore {

enum class TraceMode {
  kNone,
  kSummary,
  kVerbose,
  kJson,
  kChrome,
};

struct ObservabilityOptions {
  bool debug_enabled = false;
  TraceMode trace_mode = TraceMode::kNone;
  std::string session_id;
  std::string output_dir;
  bool force_recompile = false;
  std::size_t max_snapshot_bytes = 8 * 1024 * 1024;
  std::size_t max_total_bytes = 256 * 1024 * 1024;
};

enum class TraceEventKind {
  kCompileStart,
  kCompileEnd,
  kCacheHit,
  kCacheMiss,
  kPassStageStart,
  kPassStageEnd,
  kGpuPreflight,
  kGpuModuleLoad,
  kGpuKernelLaunch,
  kGpuKernelComplete,
  kTensorViewBind,
};

inline constexpr TraceEventKind kCompileStart = TraceEventKind::kCompileStart;
inline constexpr TraceEventKind kCompileEnd = TraceEventKind::kCompileEnd;
inline constexpr TraceEventKind kCacheHit = TraceEventKind::kCacheHit;
inline constexpr TraceEventKind kCacheMiss = TraceEventKind::kCacheMiss;
inline constexpr TraceEventKind kPassStageStart = TraceEventKind::kPassStageStart;
inline constexpr TraceEventKind kPassStageEnd = TraceEventKind::kPassStageEnd;
inline constexpr TraceEventKind kGpuPreflight = TraceEventKind::kGpuPreflight;
inline constexpr TraceEventKind kGpuModuleLoad = TraceEventKind::kGpuModuleLoad;
inline constexpr TraceEventKind kGpuKernelLaunch = TraceEventKind::kGpuKernelLaunch;
inline constexpr TraceEventKind kGpuKernelComplete =
    TraceEventKind::kGpuKernelComplete;
inline constexpr TraceEventKind kTensorViewBind = TraceEventKind::kTensorViewBind;

struct TraceEvent {
  TraceEventKind kind = TraceEventKind::kCompileStart;
  std::string name;
  std::chrono::steady_clock::time_point timestamp;
  std::string metadata;
};

class ObservabilityContext {
 public:
  class TraceScope {
   public:
    TraceScope(ObservabilityContext &ctx, TraceEventKind begin_kind,
               TraceEventKind end_kind, std::string name);
    ~TraceScope();

    TraceScope(const TraceScope &) = delete;
    TraceScope &operator=(const TraceScope &) = delete;

   private:
    ObservabilityContext &ctx_;
    TraceEventKind end_kind_;
    std::string name_;
  };

  explicit ObservabilityContext(ObservabilityOptions options);
  ~ObservabilityContext();
  ObservabilityContext(ObservabilityContext &&) noexcept;
  ObservabilityContext &operator=(ObservabilityContext &&) noexcept;

  ObservabilityContext(const ObservabilityContext &) = delete;
  ObservabilityContext &operator=(const ObservabilityContext &) = delete;

  bool snapshot(const std::string &stage_name, mlir::ModuleOp module);
  bool snapshotText(const std::string &stage_name, const std::string &content,
                    const std::string &extension = ".txt");
  void recordCacheHit(const std::string &cache_key);
  void finalize();
  void traceEvent(TraceEventKind kind, const std::string &name = "",
                  const std::string &metadata = "");
  TraceScope scopedTrace(TraceEventKind begin_kind, TraceEventKind end_kind,
                         const std::string &name = "");
  bool debugEnabled() const;
  bool traceEnabled() const;
  TraceMode traceMode() const;
  bool forceRecompile() const;
  const ObservabilityOptions &options() const;
  const std::vector<TraceEvent> &events() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void attachObservability(mlir::PassManager &pm, ObservabilityContext *ctx,
                         const std::string &stage_prefix);
ObservabilityOptions observabilityOptionsFromEnv();
ObservabilityOptions mergeObservabilityOptions(
    const ObservabilityOptions &python_opts);

}  // namespace matcore
