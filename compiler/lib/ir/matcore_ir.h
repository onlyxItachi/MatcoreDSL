#ifndef MATCORE_MDSLC_MATCORE_IR_H
#define MATCORE_MDSLC_MATCORE_IR_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace matcore::mdslc::ir {

inline constexpr std::uint32_t kMatcoreIrVersion = 0;

struct SourceLocation {
  std::string file;
  std::uint64_t offset = 0;
  std::uint32_t line = 0;
  std::uint32_t column = 0;
};

// Half-open byte offsets into the original .mdsl file.
struct SourceRange {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

struct MatrixValue {
  std::string role;
  std::string expression;
  std::string mutability;
};

struct Operation {
  std::string site_id;
  std::string kind;
  std::string canonical_callee;
  SourceLocation source;
  SourceRange call_range;
  std::vector<SourceRange> argument_ranges;
  MatrixValue output;
  std::vector<MatrixValue> operands;
  std::string target;
  std::string fallback;
  std::int64_t static_m = 0;
  std::int64_t static_n = 0;
  std::int64_t static_k = 0;
};

struct Module {
  std::string translation_unit;
  std::string source_file;
  std::string producer;
  std::vector<Operation> operations;
};

// Verifies the complete bootstrap contract before it crosses the frontend/IR
// boundary. The first diagnostic is returned in error.
bool verify(const Module &module, std::string &error);

// Serialization has a fixed key order and always ends in one newline.
std::string serializeDeterministicJson(const Module &module);

// Parses serialized bootstrap IR and runs the same verifier as the frontend.
bool parseAndVerifyJson(std::string_view json, Module &module,
                        std::string &error);

} // namespace matcore::mdslc::ir

#endif // MATCORE_MDSLC_MATCORE_IR_H
