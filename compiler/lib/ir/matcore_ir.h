#ifndef MATCORE_MDSLC_MATCORE_IR_H
#define MATCORE_MDSLC_MATCORE_IR_H

#include <cstdint>
#include <string>
#include <vector>

namespace matcore::mdslc::ir {

inline constexpr std::uint32_t kMatcoreIrVersion = 0;

struct SourceLocation {
  std::string file;
  std::uint64_t offset = 0;
  std::uint32_t line = 0;
  std::uint32_t column = 0;
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
  MatrixValue output;
  std::vector<MatrixValue> operands;
  std::string target;
  std::string fallback;
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

} // namespace matcore::mdslc::ir

#endif // MATCORE_MDSLC_MATCORE_IR_H
