# MLIR 18.1.3 Debug / Diagnostic API Reference for MatcoreDSL

This document is a reference for implementing MatcoreDSL debug infrastructure on top of MLIR 18.1.3.

## What MLIR gives you vs. what MatcoreDSL must build

MLIR already provides:

- diagnostic registration and interception via `DiagnosticEngine`
- pass lifecycle hooks via `PassInstrumentation`
- IR printing via `Operation::print`, `OpPrintingFlags`, and `AsmState`
- human-readable pass timing/statistics reports via `PassManager`

MatcoreDSL still needs to build:

- a stable structured diagnostic schema
- correlation between diagnostics, pass failures, manual transform stages, and JIT/runtime stages
- snapshot policy and artifact layout
- Chrome Trace JSON emission
- route/stage naming conventions matching `src/lowering_pipeline.cpp` and `src/jit_runner.cpp`

---

## 1. DiagnosticEngine API

### Exact MLIR 18.1.3 API signatures

From `mlir/IR/Diagnostics.h`:

```cpp
namespace mlir {
enum class DiagnosticSeverity { Note, Warning, Error, Remark };

class Diagnostic {
public:
  DiagnosticSeverity getSeverity() const;
  Location getLocation() const;
  MutableArrayRef<DiagnosticArgument> getArguments();
  ArrayRef<DiagnosticArgument> getArguments() const;
  void print(raw_ostream &os) const;
  std::string str() const;
  Diagnostic &attachNote(std::optional<Location> noteLoc = std::nullopt);
  iterator_range<const_note_iterator> getNotes() const;
};

class DiagnosticEngine {
public:
  using HandlerTy = llvm::unique_function<LogicalResult(Diagnostic &)>;
  using HandlerID = uint64_t;

  HandlerID registerHandler(HandlerTy handler);

  template <typename FuncTy, typename RetT =
      decltype(std::declval<FuncTy>()(std::declval<Diagnostic &>()))>
  std::enable_if_t<std::is_same<RetT, void>::value, HandlerID>
  registerHandler(FuncTy &&handler);

  void eraseHandler(HandlerID id);
  InFlightDiagnostic emit(Location loc, DiagnosticSeverity severity);
  void emit(Diagnostic &&diag);
};

InFlightDiagnostic emitError(Location loc);
InFlightDiagnostic emitError(Location loc, const Twine &message);
InFlightDiagnostic emitWarning(Location loc);
InFlightDiagnostic emitWarning(Location loc, const Twine &message);
InFlightDiagnostic emitRemark(Location loc);
InFlightDiagnostic emitRemark(Location loc, const Twine &message);

class ScopedDiagnosticHandler {
public:
  explicit ScopedDiagnosticHandler(MLIRContext *ctx);
  template <typename FuncTy>
  ScopedDiagnosticHandler(MLIRContext *ctx, FuncTy &&handler);
  ~ScopedDiagnosticHandler();
};
} // namespace mlir
```

### Behavior that matters

- Handlers run in stack order: newest handler first.
- Returning `success()` means the diagnostic is consumed.
- Returning `failure()` propagates it to earlier/default handlers.
- `pm.run(op)` only returns `failure()`; it does **not** tell you which pass failed.
- A pass may call `signalPassFailure()` **without** emitting a diagnostic.
- Therefore, pass-failure attribution requires **both**:
  - `DiagnosticEngine` capture
  - `PassInstrumentation::runAfterPassFailed(Pass *, Operation *)`

### Structured diagnostic capture example

```cpp
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

struct StructuredDiagArg {
  std::string kind;
  std::string text;
};

struct StructuredDiag {
  std::string severity;
  std::string location;
  std::string rendered;
  std::vector<StructuredDiagArg> arguments;
  std::vector<std::string> notes;
};

static std::string toString(mlir::DiagnosticSeverity severity) {
  switch (severity) {
  case mlir::DiagnosticSeverity::Note: return "note";
  case mlir::DiagnosticSeverity::Warning: return "warning";
  case mlir::DiagnosticSeverity::Error: return "error";
  case mlir::DiagnosticSeverity::Remark: return "remark";
  }
  return "unknown";
}

static std::string renderLocation(mlir::Location loc) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << loc;
  os.flush();
  return s;
}

static std::string renderArgument(const mlir::DiagnosticArgument &arg) {
  std::string s;
  llvm::raw_string_ostream os(s);
  arg.print(os);
  os.flush();
  return s;
}

static std::string renderDiagnostic(const mlir::Diagnostic &diag) {
  std::string s;
  llvm::raw_string_ostream os(s);
  diag.print(os);
  os.flush();
  return s;
}

static const char *argumentKindName(mlir::DiagnosticArgument::DiagnosticArgumentKind k) {
  switch (k) {
  case mlir::DiagnosticArgument::DiagnosticArgumentKind::Attribute: return "attribute";
  case mlir::DiagnosticArgument::DiagnosticArgumentKind::Double: return "double";
  case mlir::DiagnosticArgument::DiagnosticArgumentKind::Integer: return "integer";
  case mlir::DiagnosticArgument::DiagnosticArgumentKind::String: return "string";
  case mlir::DiagnosticArgument::DiagnosticArgumentKind::Type: return "type";
  case mlir::DiagnosticArgument::DiagnosticArgumentKind::Unsigned: return "unsigned";
  }
  return "unknown";
}

static std::vector<StructuredDiag>
captureDiagnosticsForRun(mlir::MLIRContext *ctx, llvm::function_ref<mlir::LogicalResult()> run) {
  std::vector<StructuredDiag> out;

  mlir::ScopedDiagnosticHandler handler(ctx, [&](mlir::Diagnostic &diag) {
    StructuredDiag item;
    item.severity = toString(diag.getSeverity());
    item.location = renderLocation(diag.getLocation());
    item.rendered = renderDiagnostic(diag);

    for (const mlir::DiagnosticArgument &arg : diag.getArguments()) {
      item.arguments.push_back({argumentKindName(arg.getKind()), renderArgument(arg)});
    }
    for (const mlir::Diagnostic &note : diag.getNotes()) {
      item.notes.push_back(renderDiagnostic(note));
    }

    out.push_back(std::move(item));
    return mlir::success();
  });

  (void)run();
  return out;
}
```

### Emitting structured diagnostics from your own code

```cpp
if (unsupportedCondition) {
  auto diag = mlir::emitError(op.getLoc(), "unsupported lowering case");
  diag << ": route=" << routeName << ", stage=" << stageName;
  diag.attachNote() << "kernel=" << kernelName;
  return mlir::failure();
}
```

### Important caveat

`Diagnostic::getArguments()` is partially structured, but many streamed values are effectively stringified at emission time. Capture structure immediately; do not promise perfect round-tripping.

---

## 2. PassInstrumentation API

### Exact MLIR 18.1.3 API signatures

From `mlir/Pass/PassInstrumentation.h` and `mlir/Pass/PassManager.h`:

```cpp
namespace mlir {
class PassInstrumentation {
public:
  struct PipelineParentInfo {
    uint64_t parentThreadID;
    Pass *parentPass;
  };

  virtual ~PassInstrumentation() = 0;

  virtual void runBeforePipeline(std::optional<OperationName> name,
                                 const PipelineParentInfo &parentInfo);
  virtual void runAfterPipeline(std::optional<OperationName> name,
                                const PipelineParentInfo &parentInfo);
  virtual void runBeforePass(Pass *pass, Operation *op) {}
  virtual void runAfterPass(Pass *pass, Operation *op) {}
  virtual void runAfterPassFailed(Pass *pass, Operation *op) {}
  virtual void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) {}
  virtual void runAfterAnalysis(StringRef name, TypeID id, Operation *op) {}
};

class PassManager : public OpPassManager {
public:
  void addInstrumentation(std::unique_ptr<PassInstrumentation> pi);
};
} // namespace mlir
```

### Registration pattern

```cpp
mlir::PassManager pm(ctx);
pm.addInstrumentation(std::make_unique<MyInstrumentation>(...));
```

### Why this is mandatory for a debug protocol

Diagnostics tell you **what** MLIR complained about.
`runAfterPassFailed` tells you **which pass failed**.
Use both.

### PassInstrumentation example: snapshot IR before/after each pass

```cpp
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

class SnapshotInstrumentation final : public mlir::PassInstrumentation {
public:
  explicit SnapshotInstrumentation(std::string rootDir)
      : rootDir(std::move(rootDir)) {}

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    writeSnapshot("before", pass, op, /*failed=*/false);
  }

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    writeSnapshot("after", pass, op, /*failed=*/false);
  }

  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    writeSnapshot("after", pass, op, /*failed=*/true);
  }

private:
  void writeSnapshot(llvm::StringRef phase, mlir::Pass *pass, mlir::Operation *op,
                     bool failed) {
    std::string path = (rootDir + "/" + phase + "-" + sanitize(pass->getName()) +
                        (failed ? "-failed" : "") + ".mlir");

    std::error_code ec;
    llvm::raw_fd_ostream file(path, ec, llvm::sys::fs::OF_Text);
    if (ec)
      return;

    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm()
         .assumeVerified()
         .elideLargeElementsAttrs()
         .elideLargeResourceString()
         .enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);

    // If the pass is not operating at module scope, local names are usually easier to read.
    if (op->getBlock())
      flags.useLocalScope();

    op->print(file, flags);
  }

  static std::string sanitize(llvm::StringRef s) {
    std::string out = s.str();
    for (char &c : out)
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-')
        c = '_';
    return out;
  }

  std::string rootDir;
};
```

### Threading caveat

If debug mode requires deterministic snapshot ordering or module-scope printing, disable MLIR multithreading or use a thread-safe collector with explicit ordering. Do **not** model structured capture as plain appends to one shared `std::string` in a multithreaded pipeline.

---

## 3. MLIR IR printing

### Exact MLIR 18.1.3 API signatures

From `mlir/IR/Operation.h`, `mlir/IR/AsmState.h`, and `mlir/IR/OperationSupport.h`:

```cpp
namespace mlir {
class Operation {
public:
  void print(raw_ostream &os, const OpPrintingFlags &flags = std::nullopt);
  void print(raw_ostream &os, AsmState &state);
  void dump();
};

class AsmState {
public:
  using LocationMap = DenseMap<Operation *, std::pair<unsigned, unsigned>>;

  AsmState(Operation *op,
           const OpPrintingFlags &printerFlags = OpPrintingFlags(),
           LocationMap *locationMap = nullptr,
           FallbackAsmResourceMap *map = nullptr);
  AsmState(MLIRContext *ctx,
           const OpPrintingFlags &printerFlags = OpPrintingFlags(),
           LocationMap *locationMap = nullptr,
           FallbackAsmResourceMap *map = nullptr);

  const OpPrintingFlags &getPrinterFlags() const;
  void attachResourcePrinter(std::unique_ptr<AsmResourcePrinter> printer);
};

class OpPrintingFlags {
public:
  OpPrintingFlags();
  OpPrintingFlags &elideLargeElementsAttrs(int64_t largeElementLimit = 16);
  OpPrintingFlags &printLargeElementsAttrWithHex(int64_t largeElementLimit = 100);
  OpPrintingFlags &elideLargeResourceString(int64_t largeResourceLimit = 64);
  OpPrintingFlags &enableDebugInfo(bool enable = true, bool prettyForm = false);
  OpPrintingFlags &printGenericOpForm(bool enable = true);
  OpPrintingFlags &skipRegions(bool skip = true);
  OpPrintingFlags &assumeVerified(bool enable = true);
  OpPrintingFlags &useLocalScope(bool enable = true);
  OpPrintingFlags &printValueUsers(bool enable = true);
  OpPrintingFlags &printUniqueSSAIDs(bool enable = true);
  OpPrintingFlags &printNameLocAsPrefix(bool enable = true);
};
} // namespace mlir
```

### Practical notes

- There is no `module.print("path")` convenience API.
- Print to a stream (`raw_string_ostream`, `raw_fd_ostream`, `errs()`, etc.).
- `AsmState` is useful when printing the same IR repeatedly and when you want a `LocationMap`.
- For failure snapshots, prefer resilient flags such as `printGenericOpForm()` and `assumeVerified()`.

### Dump module to string

```cpp
std::string dumpModuleToString(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo()
       .printUniqueSSAIDs()
       .elideLargeElementsAttrs();

  module.print(os, flags);
  os.flush();
  return text;
}
```

### Dump module to file with OpPrintingFlags

```cpp
void writeModuleToFile(mlir::ModuleOp module, llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    throw std::runtime_error("failed to open MLIR dump file");

  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm()
       .enableDebugInfo()
       .printUniqueSSAIDs()
       .elideLargeElementsAttrs()
       .elideLargeResourceString();

  module.print(file, flags);
}
```

### Reuse AsmState across prints

```cpp
std::string dumpWithAsmState(mlir::ModuleOp module) {
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo().printUniqueSSAIDs();

  mlir::AsmState::LocationMap locations;
  mlir::AsmState state(module.getOperation(), flags, &locations);

  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os, state);
  os.flush();

  // 'locations' now maps printed line/column positions back to operations.
  return text;
}
```

### Built-in IR printing via PassManager

MLIR 18.1.3 also provides:

```cpp
void PassManager::enableIRPrinting(
    std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
    std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
    bool printModuleScope = true,
    bool printAfterOnlyOnChange = true,
    bool printAfterOnlyOnFailure = false,
    raw_ostream &out = llvm::errs(),
    OpPrintingFlags opPrintingFlags = OpPrintingFlags());
```

This is useful for ad hoc debugging, but a custom `PassInstrumentation` is better when MatcoreDSL needs stable file names, JSON metadata, trace correlation, and different policies for pass vs. manual stages.

---

## 4. Pass timing and statistics

### Exact MLIR 18.1.3 API signatures

From `mlir/Pass/PassManager.h`:

```cpp
namespace mlir {
enum class PassDisplayMode {
  List,
  Pipeline,
};

class PassManager : public OpPassManager {
public:
  void enableTiming(TimingScope &timingScope);
  void enableTiming(std::unique_ptr<TimingManager> tm);
  void enableTiming();
  void enableStatistics(PassDisplayMode displayMode = PassDisplayMode::Pipeline);
};
} // namespace mlir
```

### What these APIs actually provide

- `enableTiming()` installs timing instrumentation.
- `enableStatistics()` prints collected pass statistics after `run()`.
- The built-in outputs are human-readable reports.
- They are **not** a machine-readable pass-event feed and not a Chrome Trace replacement.
- For Chrome Trace, prefer custom `PassInstrumentation` with your own clock and JSON writer.

### Minimal timing/statistics usage

```cpp
mlir::PassManager pm(ctx);
pm.enableTiming();
pm.enableStatistics(mlir::PassDisplayMode::Pipeline);

// Add passes...
if (mlir::failed(pm.run(module))) {
  // diagnostics + runAfterPassFailed should handle attribution
}
```

### Timing with an external TimingManager

```cpp
auto tm = std::make_unique<mlir::DefaultTimingManager>();
tm->setEnabled(true);
pm.enableTiming(std::move(tm));
```

### Recommendation for MatcoreDSL

Use MLIR timing/statistics as supplemental console/debug reports, but emit the debug protocol trace with custom spans:

- route-level span
- stage-level span
- per-pass span
- manual-transform span
- JIT/cache/runtime spans

---

## 5. Chrome Trace JSON format

### Format summary

For `chrome://tracing` / Perfetto-compatible JSON, each event object commonly contains:

- `name`: event name
- `cat`: category
- `ph`: phase
- `ts`: timestamp in microseconds
- `pid`: process id
- `tid`: thread id
- `args`: optional arbitrary object
- `dur`: duration in microseconds for complete (`X`) events

### Relevant event phases

- `B`: begin event
- `E`: end event
- `X`: complete event with `dur`

### Rules that matter

- Timestamps are conventionally microseconds.
- `B`/`E` pairs must be properly nested per thread.
- `X` is simpler and is recommended for MatcoreDSL pass/manual/JIT spans.
- Perfetto can open a raw JSON array of event objects; for compatibility with `chrome://tracing`, emit the conservative wrapper form with `traceEvents`.

### Recommended MatcoreDSL file shape

```json
{
  "traceEvents": [
    {
      "name": "nvidia-tensor-bufferize",
      "cat": "matcore.lowering.stage",
      "ph": "X",
      "ts": 0,
      "dur": 1287,
      "pid": 1,
      "tid": 42,
      "args": {
        "route": "nvidia-dgpu",
        "stage_index": 0,
        "snapshot_before": "artifacts/000-before.mlir",
        "snapshot_after": "artifacts/000-after.mlir"
      }
    },
    {
      "name": "Canonicalizer",
      "cat": "matcore.lowering.pass",
      "ph": "X",
      "ts": 14,
      "dur": 201,
      "pid": 1,
      "tid": 42,
      "args": {
        "stage": "nvidia-tensor-bufferize",
        "op": "builtin.module"
      }
    },
    {
      "name": "ExecutionEngine::create",
      "cat": "matcore.jit",
      "ph": "X",
      "ts": 4000,
      "dur": 921,
      "pid": 1,
      "tid": 42,
      "args": {
        "target": "x86-avx2",
        "cache_enabled": true
      }
    }
  ],
  "displayTimeUnit": "ms"
}
```

### B/E example

```json
{
  "traceEvents": [
    {"name":"nvidia-nvvm","cat":"matcore.lowering.stage","ph":"B","ts":1000,"pid":1,"tid":42},
    {"name":"nvidia-nvvm","cat":"matcore.lowering.stage","ph":"E","ts":1900,"pid":1,"tid":42}
  ]
}
```

### Suggested category names

- `matcore.lowering.route`
- `matcore.lowering.stage`
- `matcore.lowering.pass`
- `matcore.lowering.manual_stage`
- `matcore.jit`
- `matcore.cache`
- `matcore.runtime`
- `matcore.diagnostic`

---

## 6. Integration notes for MatcoreDSL

## 6.1 `src/lowering_pipeline.cpp`

Current behavior:

- NVIDIA lowering already runs as named stages inside `runLoweringPipeline(...)`:
  - `nvidia-tensor-bufferize`
  - `nvidia-dynamic-macro-topology`
  - `nvidia-post-transform-canonicalize`
  - `nvidia-mma-preparation`
  - `nvidia-launch-config`
  - `nvidia-loop-materialization`
  - `nvidia-vector-to-gpu`
  - `nvidia-nvvm`
- Generic/AMD lowering uses one `PassManager` via `configureLoweringPipeline(...)`.
- Current diagnostic handling uses `ScopedDiagnosticHandler` to append rendered text to one `std::string`.
- Existing IR fallback uses `DumpModuleIR(module)` in `src/gpu_nvvm_lowering.cpp`.

### Recommended mapping

1. Replace or wrap the current string-only diagnostic capture with a structured collector.
2. For each `PassManager` stage, install:
   - `SnapshotInstrumentation`
   - `FailureAttributionInstrumentation`
   - optional `ChromeTraceInstrumentation`
3. Capture both:
   - diagnostics emitted during `pm.run(module)`
   - the pass name from `runAfterPassFailed`
4. Emit trace spans at two levels:
   - one stage span per `run_stage(...)`
   - one pass span per `runBeforePass` / `runAfterPass`

### Critical separation: pass-managed vs. manual stages

Not everything in the NVIDIA path goes through `pm.run(...)`.
These manual transforms must get explicit begin/end/snapshot/diagnostic wrappers of their own:

- `ApplyNvidiaMmaTransformToModule(...)`
- `ApplyNvidiaMmaRewriteToModule(...)`
- `ApplyNvidiaThreadMappingToModule(...)`
- `VerifyNoResidualNvidiaMatmulOnModule(...)`

So the protocol should model two stage kinds:

- `kind = "pass-manager-stage"`
- `kind = "manual-stage"`

### Recommended per-stage record

```json
{
  "route": "nvidia-dgpu",
  "stage": "nvidia-vector-to-gpu",
  "kind": "pass-manager-stage",
  "status": "ok",
  "pass_failures": [],
  "diagnostics": [...],
  "snapshot_before": "...",
  "snapshot_after": "...",
  "trace_event_ids": [...]
}
```

### Recommended failure policy

On pass failure:

- store the failing pass name from `runAfterPassFailed`
- store all diagnostics captured during the run
- emit a failure snapshot with resilient flags:
  - `printGenericOpForm()`
  - `assumeVerified()`
  - `elideLargeElementsAttrs()`
  - `elideLargeResourceString()`
- keep `DumpModuleIR(module)` only as a last-ditch textual fallback

### Suggested implementation sketch for `run_stage(...)`

```cpp
struct StageDebugResult {
  std::string route;
  std::string stage;
  std::vector<StructuredDiag> diagnostics;
  std::optional<std::string> failingPass;
  std::string beforeSnapshot;
  std::string afterSnapshot;
};

StageDebugResult runStageWithDebug(mlir::ModuleOp module,
                                   llvm::StringRef route,
                                   llvm::StringRef stage,
                                   llvm::function_ref<void(mlir::PassManager &)> configure) {
  mlir::PassManager pm(module.getContext());
  configure(pm);

  StageDebugResult result;
  result.route = route.str();
  result.stage = stage.str();

  pm.addInstrumentation(std::make_unique<SnapshotInstrumentation>(...));
  pm.addInstrumentation(std::make_unique<FailureAttributionInstrumentation>(result));
  pm.addInstrumentation(std::make_unique<ChromeTraceInstrumentation>(...));

  result.diagnostics = captureDiagnosticsForRun(module.getContext(), [&] {
    return pm.run(module);
  });
  return result;
}
```

## 6.2 `src/jit_runner.cpp`

The debug protocol should continue past lowering.
Important debug boundaries in `jit_runner.cpp`:

- disk-cache lookup / `dlopen` path
- `MlirEngine::BuildAndLower(...)`
- `ExecutionEngine::create(...)`
- object dump / shared-library persistence
- runtime symbol registration
- packed invocation / runtime failure surface

### Recommended trace spans in JIT/runtime

Emit `X` events for at least:

- `tryLoadDiskCachedExecution`
- `MlirEngine::BuildAndLower`
- `ExecutionEngine::create`
- `registerExecutionRuntimeSymbols`
- `registerGpuRuntimeSymbols`
- `persistExecutionToDiskCache`
- `invokePacked`
- top-level `compileAndRun`

### Recommended JIT debug artifacts

- final lowered MLIR module before `ExecutionEngine::create`
- object-file path if `dumpToObjectFile(...)` is enabled
- cache hit/miss metadata
- shared library path for disk cache artifacts
- runtime invocation error string on `invokePacked` failure

### Practical mapping

- Lowering protocol artifacts belong to the build/lowering phase.
- JIT protocol artifacts belong to compilation/execution phase.
- Keep the same correlation id across both so one front-end request maps to one lowering trace plus one JIT/runtime trace.

---

## 7. Recommended MatcoreDSL implementation choices

1. **Use `ScopedDiagnosticHandler` for lifetime management**.
2. **Use `PassInstrumentation` for pass attribution and spans**.
3. **Use manual wrappers for non-pass NVIDIA transform stages**.
4. **Prefer `X` events for Chrome Trace**.
5. **Use resilient printing flags on failure snapshots**.
6. **Treat `enableTiming()` / `enableStatistics()` as supplemental reports, not the main protocol**.
7. **In debug mode, strongly consider disabling MLIR multithreading** unless deterministic ordering is implemented explicitly.

---

## References

Primary MLIR 18.1.3 sources used for this document:

- `llvm-project` tag `llvmorg-18.1.3`
  - `mlir/include/mlir/IR/Diagnostics.h`
  - `mlir/docs/Diagnostics.md`
  - `mlir/include/mlir/Pass/PassInstrumentation.h`
  - `mlir/include/mlir/Pass/PassManager.h`
  - `mlir/docs/PassManagement.md`
  - `mlir/include/mlir/IR/Operation.h`
  - `mlir/include/mlir/IR/AsmState.h`
  - `mlir/lib/Pass/IRPrinting.cpp`
  - `mlir/lib/Pass/PassTiming.cpp`
  - `mlir/lib/Pass/PassStatistics.cpp`
- MLIR doxygen pages:
  - `mlir::DiagnosticEngine`
  - `mlir::PassManager`
  - `mlir::PassInstrumentation`
  - `mlir::AsmState`
  - `mlir::OpPrintingFlags`
- Chrome / Perfetto trace references:
  - Perfetto: external Chrome JSON trace format
  - Chromium trace event profiling tool documentation
  - Chromium trace event best practices

Repo integration context:

- `/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp`
- `/home/hamza-usta/MatcoreDSL/src/gpu_nvvm_lowering.cpp`
- `/home/hamza-usta/MatcoreDSL/src/jit_runner.cpp`
