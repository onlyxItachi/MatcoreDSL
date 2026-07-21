#include "matcore/observability.h"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <memory>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"

namespace matcore {
namespace {

std::string ToLower(std::string value) {
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

std::string SanitizePathToken(const std::string &value,
                              const std::string &fallback) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    const bool keep =
        (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
    out.push_back(keep ? ch : '_');
  }
  if (out.empty()) {
    return fallback;
  }
  return out;
}

bool ParseBool(const char *value) {
  if (value == nullptr) {
    return false;
  }
  const std::string lowered = ToLower(value);
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on";
}

TraceMode ParseTraceMode(const std::string &value) {
  const std::string lowered = ToLower(value);
  if (lowered == "summary") {
    return TraceMode::kSummary;
  }
  if (lowered == "verbose") {
    return TraceMode::kVerbose;
  }
  if (lowered == "json") {
    return TraceMode::kJson;
  }
  if (lowered == "chrome") {
    return TraceMode::kChrome;
  }
  return TraceMode::kNone;
}

const char *TraceModeName(TraceMode mode) {
  switch (mode) {
    case TraceMode::kNone:
      return "none";
    case TraceMode::kSummary:
      return "summary";
    case TraceMode::kVerbose:
      return "verbose";
    case TraceMode::kJson:
      return "json";
    case TraceMode::kChrome:
      return "chrome";
  }
  return "none";
}

const char *TraceEventKindName(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::kCompileStart:
      return "compile_start";
    case TraceEventKind::kCompileEnd:
      return "compile_end";
    case TraceEventKind::kCacheHit:
      return "cache_hit";
    case TraceEventKind::kCacheMiss:
      return "cache_miss";
    case TraceEventKind::kPassStageStart:
      return "pass_stage_start";
    case TraceEventKind::kPassStageEnd:
      return "pass_stage_end";
    case TraceEventKind::kGpuPreflight:
      return "gpu_preflight";
    case TraceEventKind::kGpuModuleLoad:
      return "gpu_module_load";
    case TraceEventKind::kGpuKernelLaunch:
      return "gpu_kernel_launch";
    case TraceEventKind::kGpuKernelComplete:
      return "gpu_kernel_complete";
    case TraceEventKind::kTensorViewBind:
      return "tensor_view_bind";
  }
  return "unknown";
}

std::string NowIso8601() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
#if defined(_WIN32)
  gmtime_s(&tm_buf, &now_time);
#else
  gmtime_r(&now_time, &tm_buf);
#endif
  std::ostringstream stream;
  stream << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

std::string Iso8601FromTimePoint(std::chrono::system_clock::time_point tp) {
  const std::time_t time_value = std::chrono::system_clock::to_time_t(tp);
  std::tm tm_buf;
#if defined(_WIN32)
  gmtime_s(&tm_buf, &time_value);
#else
  gmtime_r(&time_value, &tm_buf);
#endif
  std::ostringstream stream;
  stream << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

std::string ToJsonString(const llvm::json::Value &value) {
  return llvm::formatv("{0:2}", value).str();
}

char ChromePhaseForKind(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::kCompileStart:
    case TraceEventKind::kPassStageStart:
    case TraceEventKind::kGpuKernelLaunch:
      return 'B';
    case TraceEventKind::kCompileEnd:
    case TraceEventKind::kPassStageEnd:
    case TraceEventKind::kGpuKernelComplete:
      return 'E';
    case TraceEventKind::kCacheHit:
    case TraceEventKind::kCacheMiss:
    case TraceEventKind::kGpuPreflight:
    case TraceEventKind::kGpuModuleLoad:
    case TraceEventKind::kTensorViewBind:
      return 'i';
  }
  return 'i';
}

bool IsTraceFileMode(TraceMode mode) {
  return mode == TraceMode::kJson || mode == TraceMode::kChrome;
}

bool WriteTextFile(const std::filesystem::path &path, const std::string &content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }
  out << content;
  return out.good();
}

std::optional<std::int64_t> ComputeCompileDurationMs(
    const std::vector<TraceEvent> &events) {
  std::optional<std::chrono::steady_clock::time_point> compile_start;
  for (const TraceEvent &event : events) {
    if (event.kind == TraceEventKind::kCompileStart && !compile_start.has_value()) {
      compile_start = event.timestamp;
      continue;
    }
    if (event.kind == TraceEventKind::kCompileEnd && compile_start.has_value()) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 event.timestamp - *compile_start)
          .count();
    }
  }
  return std::nullopt;
}

std::string ComputeCacheStatus(const std::vector<TraceEvent> &events) {
  for (const TraceEvent &event : events) {
    if (event.kind == TraceEventKind::kCacheHit) {
      return "hit";
    }
    if (event.kind == TraceEventKind::kCacheMiss) {
      return "miss";
    }
  }
  return "none";
}

std::size_t CountPassStages(const std::vector<TraceEvent> &events) {
  std::size_t count = 0;
  for (const TraceEvent &event : events) {
    if (event.kind == TraceEventKind::kPassStageStart) {
      ++count;
    }
  }
  return count;
}

}  // namespace

struct ObservabilityContext::Impl {
  explicit Impl(ObservabilityOptions in_options)
      : options(std::move(in_options)), start_time(std::chrono::steady_clock::now()) {
    options.force_recompile = options.force_recompile || options.debug_enabled;

    std::string session = SanitizePathToken(options.session_id, "session");
    if (options.output_dir.empty()) {
      output_dir = std::filesystem::path(".matcore_debug") / session;
      options.output_dir = output_dir.string();
    } else {
      output_dir = std::filesystem::path(options.output_dir);
    }

    const bool needs_output_dir = options.debug_enabled || options.trace_mode != TraceMode::kNone;
    output_ready = !needs_output_dir;
    if (needs_output_dir) {
      std::error_code ec;
      std::filesystem::create_directories(output_dir, ec);
      output_ready = !ec;
    }
  }

  ObservabilityOptions options;
  std::uint64_t snapshot_counter = 0;
  std::size_t total_bytes_written = 0;
  std::vector<TraceEvent> events;
  std::filesystem::path output_dir;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::system_clock::time_point wall_start_time =
      std::chrono::system_clock::now();
  bool finalized = false;
  bool output_ready = false;
  std::mutex mutex;
};

ObservabilityContext::ObservabilityContext(ObservabilityOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

ObservabilityContext::~ObservabilityContext() = default;
ObservabilityContext::ObservabilityContext(ObservabilityContext &&) noexcept = default;
ObservabilityContext &ObservabilityContext::operator=(
    ObservabilityContext &&) noexcept = default;

bool ObservabilityContext::snapshot(const std::string &stage_name,
                                    mlir::ModuleOp module) {
  if (!debugEnabled() || !module) {
    return false;
  }

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module.print(stream);
  stream.flush();
  return snapshotText(stage_name, ir, ".mlir");
}

bool ObservabilityContext::snapshotText(const std::string &stage_name,
                                        const std::string &content,
                                        const std::string &extension) {
  if (!debugEnabled()) {
    return false;
  }

  const std::string safe_stage = SanitizePathToken(stage_name, "stage");
  std::string safe_extension = extension.empty() ? ".txt" : extension;
  if (safe_extension.front() != '.') {
    safe_extension.insert(safe_extension.begin(), '.');
  }
  safe_extension = SanitizePathToken(safe_extension, ".txt");

  std::string output = content;
  const std::size_t max_snapshot_bytes = impl_->options.max_snapshot_bytes;
  if (output.size() > max_snapshot_bytes) {
    output.resize(max_snapshot_bytes);
    output += "\n... [TRUNCATED at " + std::to_string(max_snapshot_bytes) +
              " bytes]\n";
  }

  std::uint64_t id = 0;
  std::size_t bytes = output.size();
  std::filesystem::path output_path;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->finalized || !impl_->output_ready) {
      return false;
    }
    if (impl_->total_bytes_written + bytes > impl_->options.max_total_bytes) {
      return false;
    }
    id = ++impl_->snapshot_counter;
    impl_->total_bytes_written += bytes;

    std::ostringstream file_name;
    file_name << std::setw(2) << std::setfill('0') << id << '_' << safe_stage
              << safe_extension;
    output_path = impl_->output_dir / file_name.str();
  }

  if (!WriteTextFile(output_path, output)) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->total_bytes_written >= bytes) {
      impl_->total_bytes_written -= bytes;
    } else {
      impl_->total_bytes_written = 0;
    }
    return false;
  }

  return true;
}

void ObservabilityContext::recordCacheHit(const std::string &cache_key) {
  if (!debugEnabled()) {
    return;
  }

  llvm::json::Object payload{
      {"cache_key", cache_key},
      {"timestamp", NowIso8601()},
  };

  std::filesystem::path path;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->finalized || !impl_->output_ready) {
      return;
    }
    path = impl_->output_dir / "cache_hit.json";
  }

  WriteTextFile(path, ToJsonString(llvm::json::Value(std::move(payload))));
}

void ObservabilityContext::finalize() {
  std::vector<TraceEvent> events;
  ObservabilityOptions options;
  std::filesystem::path output_dir;
  std::uint64_t snapshots = 0;
  std::size_t bytes_written = 0;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::system_clock::time_point wall_start;
  bool output_ready = false;

  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->finalized) {
      return;
    }
    impl_->finalized = true;

    events = impl_->events;
    options = impl_->options;
    output_dir = impl_->output_dir;
    snapshots = impl_->snapshot_counter;
    bytes_written = impl_->total_bytes_written;
    start_time = impl_->start_time;
    wall_start = impl_->wall_start_time;
    output_ready = impl_->output_ready;
  }

  const auto end_time = std::chrono::steady_clock::now();
  const auto wall_end = std::chrono::system_clock::now();
  const auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
          .count();

  if (output_ready) {
    std::unordered_map<std::string, std::uint64_t> event_counts;
    for (const TraceEvent &event : events) {
      ++event_counts[TraceEventKindName(event.kind)];
    }

    llvm::json::Object counts_json;
    for (const auto &[kind, count] : event_counts) {
      counts_json[kind] = static_cast<std::int64_t>(count);
    }

    llvm::json::Object metadata{
        {"session_id", options.session_id},
        {"output_dir", options.output_dir},
        {"start_time", Iso8601FromTimePoint(wall_start)},
        {"end_time", Iso8601FromTimePoint(wall_end)},
        {"duration_ms", duration_ms},
        {"snapshot_count", static_cast<std::int64_t>(snapshots)},
        {"total_bytes_written", static_cast<std::int64_t>(bytes_written)},
        {"trace_mode", TraceModeName(options.trace_mode)},
        {"trace_event_count", static_cast<std::int64_t>(events.size())},
        {"trace_events_by_kind", std::move(counts_json)},
    };

    WriteTextFile(output_dir / "session_metadata.json",
                  ToJsonString(llvm::json::Value(std::move(metadata))));

    if (options.trace_mode == TraceMode::kJson) {
      llvm::json::Array trace_events;
      const auto origin = start_time;
      for (const TraceEvent &event : events) {
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                event.timestamp - origin)
                .count();
        trace_events.push_back(llvm::json::Object{
            {"kind", TraceEventKindName(event.kind)},
            {"name", event.name},
            {"timestamp_us", elapsed_us},
            {"metadata", event.metadata},
        });
      }
      const std::string trace_name =
          SanitizePathToken(options.session_id, "session") + "_trace.json";
      WriteTextFile(output_dir / trace_name,
                    ToJsonString(llvm::json::Value(std::move(trace_events))));
    } else if (options.trace_mode == TraceMode::kChrome) {
      llvm::json::Array trace_events;
      const auto origin = start_time;
      for (const TraceEvent &event : events) {
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                event.timestamp - origin)
                .count();
        const char phase = ChromePhaseForKind(event.kind);
        llvm::json::Object chrome_event{
            {"name", event.name.empty() ? TraceEventKindName(event.kind)
                                         : event.name},
            {"cat", "matcore"},
            {"ph", std::string(1, phase)},
            {"ts", elapsed_us},
            {"pid", 1},
            {"tid", 1},
            {"args", llvm::json::Object{{"metadata", event.metadata}}},
        };
        if (phase == 'i') {
          chrome_event["s"] = "t";
        }
        trace_events.push_back(std::move(chrome_event));
      }
      WriteTextFile(output_dir / "trace.json",
                    ToJsonString(llvm::json::Value(std::move(trace_events))));
    }
  }

  if (options.trace_mode == TraceMode::kSummary) {
    const std::optional<std::int64_t> compile_ms = ComputeCompileDurationMs(events);
    const std::string cache_status = ComputeCacheStatus(events);
    const std::size_t pass_count = CountPassStages(events);
    if (compile_ms.has_value()) {
      std::fprintf(stderr,
                   "MatCore: compile=%lldms cache=%s passes=%zu total=%lldms\n",
                   static_cast<long long>(*compile_ms), cache_status.c_str(),
                   pass_count, static_cast<long long>(duration_ms));
    } else {
      std::fprintf(stderr,
                   "MatCore: compile=n/a cache=%s passes=%zu total=%lldms\n",
                   cache_status.c_str(), pass_count,
                   static_cast<long long>(duration_ms));
    }
  } else if (options.trace_mode == TraceMode::kVerbose) {
    const auto origin = start_time;
    for (const TraceEvent &event : events) {
      const auto elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              event.timestamp - origin)
              .count();
      std::fprintf(stderr,
                   "[matcore-trace] t_us=%lld kind=%s name=%s metadata=%s\n",
                   static_cast<long long>(elapsed_us),
                   TraceEventKindName(event.kind), event.name.c_str(),
                   event.metadata.c_str());
    }
  }

  (void)wall_start;
  (void)wall_end;
}

void ObservabilityContext::traceEvent(TraceEventKind kind, const std::string &name,
                                      const std::string &metadata) {
  if (!traceEnabled()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->finalized) {
      return;
    }
    impl_->events.push_back(
        TraceEvent{kind, name, std::chrono::steady_clock::now(), metadata});
  }
}

ObservabilityContext::TraceScope::TraceScope(ObservabilityContext &ctx,
                                             TraceEventKind begin_kind,
                                             TraceEventKind end_kind,
                                             std::string name)
    : ctx_(ctx), end_kind_(end_kind), name_(std::move(name)) {
  ctx_.traceEvent(begin_kind, name_);
}

ObservabilityContext::TraceScope::~TraceScope() {
  ctx_.traceEvent(end_kind_, name_);
}

ObservabilityContext::TraceScope ObservabilityContext::scopedTrace(
    TraceEventKind begin_kind, TraceEventKind end_kind,
    const std::string &name) {
  return TraceScope(*this, begin_kind, end_kind, name);
}

bool ObservabilityContext::debugEnabled() const {
  return impl_->options.debug_enabled;
}

bool ObservabilityContext::traceEnabled() const {
  return impl_->options.trace_mode != TraceMode::kNone;
}

TraceMode ObservabilityContext::traceMode() const {
  return impl_->options.trace_mode;
}

bool ObservabilityContext::forceRecompile() const {
  return impl_->options.force_recompile;
}

const ObservabilityOptions &ObservabilityContext::options() const {
  return impl_->options;
}

const std::vector<TraceEvent> &ObservabilityContext::events() const {
  return impl_->events;
}

namespace {

class StageSnapshotInstrumentation final : public mlir::PassInstrumentation {
 public:
  StageSnapshotInstrumentation(ObservabilityContext *ctx, std::string stage_prefix)
      : ctx_(ctx), stage_prefix_(std::move(stage_prefix)) {}

  void runBeforePipeline(std::optional<mlir::OperationName>,
                         const PipelineParentInfo &) override {
    armed_before_pass_.store(true);
  }

  void runBeforePass(mlir::Pass *, mlir::Operation *op) override {
    if (!armed_before_pass_.load() || pre_snapshot_done_.exchange(true) || !op ||
        ctx_ == nullptr) {
      return;
    }

    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
      module = op->getParentOfType<mlir::ModuleOp>();
    }
    if (!module) {
      return;
    }
    module_ = module.getOperation();
    ctx_->snapshot(stage_prefix_ + "_pre", module);
  }

  void runAfterPipeline(std::optional<mlir::OperationName>,
                        const PipelineParentInfo &) override {
    if (post_snapshot_done_.exchange(true) || ctx_ == nullptr) {
      return;
    }
    if (auto *op = module_.load()) {
      if (auto module = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        ctx_->snapshot(stage_prefix_ + "_post", module);
      }
    }
  }

 private:
  ObservabilityContext *ctx_ = nullptr;
  std::string stage_prefix_;
  std::atomic<bool> armed_before_pass_{false};
  std::atomic<bool> pre_snapshot_done_{false};
  std::atomic<bool> post_snapshot_done_{false};
  std::atomic<mlir::Operation *> module_{nullptr};
};

}  // namespace

void attachObservability(mlir::PassManager &pm, ObservabilityContext *ctx,
                         const std::string &stage_prefix) {
  if (ctx == nullptr || !ctx->debugEnabled()) {
    return;
  }
  pm.addInstrumentation(
      std::make_unique<StageSnapshotInstrumentation>(ctx, stage_prefix));
}

ObservabilityOptions observabilityOptionsFromEnv() {
  ObservabilityOptions options;

  if (const char *debug = std::getenv("MATCORE_DEBUG")) {
    options.debug_enabled = ParseBool(debug);
  }
  if (const char *debug_dir = std::getenv("MATCORE_DEBUG_DIR")) {
    options.output_dir = debug_dir;
  }
  if (const char *trace = std::getenv("MATCORE_TRACE")) {
    options.trace_mode = ParseTraceMode(trace);
  }
  if (const char *session = std::getenv("MATCORE_DEBUG_SESSION")) {
    options.session_id = session;
  }

  options.force_recompile = options.debug_enabled;
  return options;
}

ObservabilityOptions mergeObservabilityOptions(
    const ObservabilityOptions &python_opts) {
  ObservabilityOptions merged = python_opts;

  if (const char *debug = std::getenv("MATCORE_DEBUG")) {
    merged.debug_enabled = ParseBool(debug);
  }
  if (const char *debug_dir = std::getenv("MATCORE_DEBUG_DIR")) {
    merged.output_dir = debug_dir;
  }
  if (const char *trace = std::getenv("MATCORE_TRACE")) {
    merged.trace_mode = ParseTraceMode(trace);
  }
  if (const char *session = std::getenv("MATCORE_DEBUG_SESSION")) {
    merged.session_id = session;
  }

  merged.force_recompile = merged.force_recompile || merged.debug_enabled;
  return merged;
}

}  // namespace matcore
