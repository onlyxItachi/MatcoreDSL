#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;
namespace dialect = matcore::mdslc::mlir_dialect;
namespace v1 = matcore::mdslc::ir::v1;

int failures = 0;
int checks = 0;

void check(bool condition, std::string_view message) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

std::string readFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  check(stream.good() || stream.eof(), "fixture must be readable");
  return contents.str();
}

v1::Module readCapture() {
  const std::string json =
      readFile(std::string(MDSLC_IR_TEST_SOURCE_DIR) +
               "/gemm_capture.v1.golden.json");
  v1::Module module;
  std::string error;
  check(v1::parseAndVerifyJson(json, module, error),
        "reviewed v1 capture must parse and verify");
  return module;
}

void makeStatic(v1::TensorValue &value, std::uint64_t rows,
                std::uint64_t columns) {
  value.type.shape = {v1::ScalarExpr::staticValue(rows),
                      v1::ScalarExpr::staticValue(columns)};
  value.type.strides = {v1::ScalarExpr::staticValue(columns),
                        v1::ScalarExpr::staticValue(1)};
}

mlir::DictionaryAttr scalarStatic(mlir::Builder &builder, std::int64_t value) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("kind", builder.getStringAttr("static")),
       builder.getNamedAttr("value", builder.getI64IntegerAttr(value))});
}

mlir::DictionaryAttr scalarDynamic(mlir::Builder &builder,
                                   llvm::StringRef symbol) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("kind", builder.getStringAttr("dynamic")),
       builder.getNamedAttr("symbol", builder.getStringAttr(symbol))});
}

mlir::DictionaryAttr tensorContract(mlir::Builder &builder,
                                    mlir::RankedTensorType type,
                                    mlir::ArrayAttr shape) {
  llvm::SmallVector<mlir::Attribute> strides;
  for (int64_t index = 1; index < type.getRank(); ++index)
    strides.push_back(shape[index]);
  strides.push_back(scalarStatic(builder, 1));
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "aliasing", builder.getStringAttr("functional_result_no_inplace")),
       builder.getNamedAttr("alignment_bytes",
                            builder.getI64IntegerAttr(4)),
       builder.getNamedAttr(
           "alignment_contract",
           builder.getStringAttr(
               "required_precondition_and_result_contract")),
       builder.getNamedAttr("dtype", builder.getStringAttr("f32")),
       builder.getNamedAttr("input_mutability",
                            builder.getStringAttr("read")),
       builder.getNamedAttr("layout",
                            builder.getStringAttr("row_major_contiguous")),
       builder.getNamedAttr("memory_space", builder.getStringAttr("host")),
       builder.getNamedAttr("rank",
                            builder.getI64IntegerAttr(type.getRank())),
       builder.getNamedAttr("result_mutability",
                            builder.getStringAttr("functional_write_once")),
       builder.getNamedAttr("shape", shape),
       builder.getNamedAttr("strides", builder.getArrayAttr(strides))});
}

mlir::DictionaryAttr mapEffects(mlir::Builder &builder, bool has_mask) {
  llvm::SmallVector<mlir::Attribute> reads = {
      builder.getStringAttr("input")};
  if (has_mask)
    reads.push_back(builder.getStringAttr("mask"));
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("read_write", builder.getArrayAttr({})),
       builder.getNamedAttr("reads", builder.getArrayAttr(reads)),
       builder.getNamedAttr("writes", builder.getArrayAttr({}))});
}

mlir::DictionaryAttr mapNumerical(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("approximate_math", builder.getBoolAttr(false)),
       builder.getNamedAttr(
           "domain_application",
           builder.getStringAttr("active_elements_only")),
       builder.getNamedAttr("inplace", builder.getBoolAttr(false)),
       builder.getNamedAttr("profile", builder.getStringAttr("map-f32-v1")),
       builder.getNamedAttr(
           "result_identity",
           builder.getStringAttr("new_functional_value"))});
}

mlir::DictionaryAttr sinNumerical(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "accuracy", builder.getStringAttr("correctly_rounded_f32")),
       builder.getNamedAttr("approximate_math", builder.getBoolAttr(false)),
       builder.getNamedAttr(
           "exception_status",
           builder.getStringAttr("postcall_unspecified")),
       builder.getNamedAttr("infinity", builder.getStringAttr("quiet_nan")),
       builder.getNamedAttr(
           "nan", builder.getStringAttr("quiet_nan_payload_unspecified")),
       builder.getNamedAttr("profile",
                            builder.getStringAttr("sin-f32-ieee-v1")),
       builder.getNamedAttr("rounding",
                            builder.getStringAttr("nearest_ties_even")),
       builder.getNamedAttr("signed_zero",
                            builder.getStringAttr("preserve")),
       builder.getNamedAttr(
           "subnormals",
           builder.getStringAttr("ieee_gradual_ftz_daz_forbidden")),
       builder.getNamedAttr("trapping_exceptions",
                            builder.getStringAttr("unsupported"))});
}

mlir::DictionaryAttr allDomain(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("kind", builder.getStringAttr("all")),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

mlir::DictionaryAttr sliceDomain(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "begin", builder.getArrayAttr({builder.getI64IntegerAttr(1),
                                           builder.getI64IntegerAttr(2)})),
       builder.getNamedAttr(
           "end", builder.getArrayAttr({builder.getI64IntegerAttr(7),
                                         builder.getI64IntegerAttr(6)})),
       builder.getNamedAttr("kind", builder.getStringAttr("slice")),
       builder.getNamedAttr(
           "step", builder.getArrayAttr({builder.getI64IntegerAttr(2),
                                          builder.getI64IntegerAttr(1)})),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

mlir::DictionaryAttr indicesDomain(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "coordinates",
           builder.getArrayAttr(
               {builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                      builder.getI64IntegerAttr(1)}),
                builder.getArrayAttr({builder.getI64IntegerAttr(7),
                                      builder.getI64IntegerAttr(6)})})),
       builder.getNamedAttr("kind", builder.getStringAttr("indices")),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

mlir::DictionaryAttr maskDomain(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("kind", builder.getStringAttr("mask")),
       builder.getNamedAttr("shape_equality",
                            builder.getStringAttr("required_precondition")),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

mlir::DictionaryAttr semanticProvenance(mlir::Builder &builder,
                                        mlir::Location location,
                                        llvm::StringRef kind,
                                        llvm::StringRef source_anchor,
                                        llvm::StringRef authenticity) {
  const auto source = mlir::cast<mlir::FileLineColLoc>(location);
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("authenticity",
                            builder.getStringAttr(authenticity)),
       builder.getNamedAttr("column",
                            builder.getI64IntegerAttr(source.getColumn())),
       builder.getNamedAttr("file",
                            builder.getStringAttr(source.getFilename().getValue())),
       builder.getNamedAttr("kind", builder.getStringAttr(kind)),
       builder.getNamedAttr("line",
                            builder.getI64IntegerAttr(source.getLine())),
       builder.getNamedAttr("source_anchor",
                            builder.getStringAttr(source_anchor)),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

mlir::DictionaryAttr sourceAuthenticatedProvenance(
    mlir::Builder &builder, mlir::Location location, llvm::StringRef kind,
    llvm::StringRef source_anchor, llvm::StringRef source_snapshot,
    std::uint64_t begin, std::uint64_t end) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (mlir::NamedAttribute attribute :
       semanticProvenance(builder, location, kind, source_anchor,
                          "source_authenticated"))
    attributes.push_back(attribute);
  attributes.push_back(builder.getNamedAttr(
      "source_range",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("begin", builder.getI64IntegerAttr(begin)),
           builder.getNamedAttr("end", builder.getI64IntegerAttr(end))})));
  attributes.push_back(builder.getNamedAttr(
      "source_snapshot", builder.getStringAttr(source_snapshot)));
  return builder.getDictionaryAttr(attributes);
}

mlir::DictionaryAttr withAddedField(mlir::Builder &builder,
                                    mlir::DictionaryAttr dictionary,
                                    llvm::StringRef field,
                                    mlir::Attribute value) {
  llvm::SmallVector<mlir::NamedAttribute> attributes(dictionary.begin(),
                                                      dictionary.end());
  attributes.push_back(builder.getNamedAttr(field, value));
  return builder.getDictionaryAttr(attributes);
}

mlir::DictionaryAttr withField(mlir::Builder &builder,
                               mlir::DictionaryAttr dictionary,
                               llvm::StringRef field,
                               mlir::Attribute replacement) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  bool replaced = false;
  for (mlir::NamedAttribute attribute : dictionary) {
    if (attribute.getName().strref() == field) {
      attributes.push_back(builder.getNamedAttr(field, replacement));
      replaced = true;
    } else {
      attributes.push_back(attribute);
    }
  }
  check(replaced, "mutated map semantic field must exist");
  return builder.getDictionaryAttr(attributes);
}

mlir::DictionaryAttr withoutField(mlir::Builder &builder,
                                  mlir::DictionaryAttr dictionary,
                                  llvm::StringRef field) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (mlir::NamedAttribute attribute : dictionary) {
    if (attribute.getName().strref() != field)
      attributes.push_back(attribute);
  }
  return builder.getDictionaryAttr(attributes);
}

dialect::MapOp appendSinMap(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::Value input, mlir::Value mask,
                            mlir::DictionaryAttr domain,
                            llvm::StringRef outside_domain,
                            mlir::ArrayAttr shape,
                            llvm::StringRef source_anchor,
                            bool derived_from_producer) {
  const auto type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto map = builder.create<dialect::MapOp>(
      location, type, input, mask, domain,
      builder.getStringAttr(outside_domain), tensorContract(builder, type, shape),
      mapEffects(builder, static_cast<bool>(mask)), mapNumerical(builder),
      semanticProvenance(
          builder, location,
          derived_from_producer ? "derived_semantic_composition"
                                : "synthetic_semantic_composition",
          source_anchor, derived_from_producer ? "derived_from_producer"
                                               : "synthetic_test_fixture"));
  auto *body = new mlir::Block();
  map.getBody().push_back(body);
  mlir::Value element = body->addArgument(builder.getF32Type(), location);
  builder.setInsertionPointToEnd(body);
  auto sin = builder.create<dialect::SinOp>(
      location, builder.getF32Type(), element, sinNumerical(builder),
      semanticProvenance(
          builder, location,
          derived_from_producer ? "derived_elementwise_expression"
                                : "synthetic_elementwise_expression",
          source_anchor, derived_from_producer ? "derived_from_producer"
                                               : "synthetic_test_fixture"));
  builder.create<dialect::YieldOp>(location, sin.getResult());
  return map;
}

bridge::BridgeResult buildGemmMap(v1::Module capture, mlir::MLIRContext &context,
                                  bool use_slice) {
  if (use_slice) {
    makeStatic(capture.operations[0].operands[0], 8, 8);
    makeStatic(capture.operations[0].operands[1], 8, 8);
    makeStatic(capture.operations[0].output, 8, 8);
  }
  auto result = bridge::bridgeV1ToMatcoreMlir(
      capture, context, bridge::explicitGemmF32V1BridgeContext());
  if (!result)
    return result;
  dialect::GemmOp gemm;
  mlir::func::ReturnOp return_op;
  result.module->walk([&](dialect::GemmOp operation) { gemm = operation; });
  result.module->walk(
      [&](mlir::func::ReturnOp operation) { return_op = operation; });
  mlir::OpBuilder builder(&context);
  builder.setInsertionPoint(return_op);
  mlir::ArrayAttr shape =
      use_slice
          ? builder.getArrayAttr({scalarStatic(builder, 8),
                                  scalarStatic(builder, 8)})
          : builder.getArrayAttr({scalarDynamic(builder, "m"),
                                  scalarDynamic(builder, "n")});
  auto map = appendSinMap(builder, gemm.getLoc(), gemm.getResult(), {},
                          use_slice ? sliceDomain(builder) : allDomain(builder),
                          use_slice ? "preserve_input" : "not_applicable",
                          shape, gemm.getSiteId(),
                          /*derived_from_producer=*/true);
  return_op.setOperand(0, map.getResult());
  mlir::Builder module_builder(&context);
  (*result.module)->setAttr(
      "mdsl.composition_schema",
      module_builder.getStringAttr(dialect::kCompositionSchemaV1));
  (*result.module)->setAttr("mdsl.composition_version",
                            module_builder.getI32IntegerAttr(1));
  return result;
}

mlir::OwningOpRef<mlir::ModuleOp>
buildStandaloneMap(mlir::MLIRContext &context, mlir::DictionaryAttr domain,
                   bool with_mask, bool dynamic_shape = false,
                   bool rank_three = false, int64_t mask_columns = 8,
                   bool encoded_tensor = false, bool encoded_mask = false) {
  mlir::OpBuilder builder(&context);
  const auto location =
      mlir::FileLineColLoc::get(&context, "map_domain_fixture.mdsl", 1, 1);
  auto module = mlir::ModuleOp::create(location);
  const auto tensor = mlir::RankedTensorType::get(
      rank_three ? llvm::ArrayRef<int64_t>{2, 3, 4}
                 : dynamic_shape ? llvm::ArrayRef<int64_t>{
                                       mlir::ShapedType::kDynamic,
                                       mlir::ShapedType::kDynamic}
                                 : llvm::ArrayRef<int64_t>{8, 8},
      builder.getF32Type(),
      encoded_tensor ? mlir::Attribute(builder.getStringAttr("test.encoding"))
                     : mlir::Attribute{});
  const auto mask = mlir::RankedTensorType::get(
      dynamic_shape
          ? llvm::ArrayRef<int64_t>{mlir::ShapedType::kDynamic,
                                    mlir::ShapedType::kDynamic}
          : llvm::ArrayRef<int64_t>{8, mask_columns},
      builder.getI1Type(),
      encoded_mask ? mlir::Attribute(builder.getStringAttr("test.encoding"))
                   : mlir::Attribute{});
  llvm::SmallVector<mlir::Type> inputs = {tensor};
  if (with_mask)
    inputs.push_back(mask);
  auto function = mlir::func::FuncOp::create(
      location, "map_domain_fixture",
      builder.getFunctionType(inputs, {tensor}));
  module.push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::Value mask_value = with_mask ? entry->getArgument(1) : mlir::Value{};
  mlir::ArrayAttr shape;
  if (rank_three)
    shape = builder.getArrayAttr({scalarStatic(builder, 2),
                                  scalarStatic(builder, 3),
                                  scalarStatic(builder, 4)});
  else if (dynamic_shape)
    shape = builder.getArrayAttr({scalarDynamic(builder, "m"),
                                  scalarDynamic(builder, "n")});
  else
    shape = builder.getArrayAttr(
        {scalarStatic(builder, 8), scalarStatic(builder, 8)});
  auto map = appendSinMap(
      builder, location, entry->getArgument(0), mask_value,
      domain, domain.getAs<mlir::StringAttr>("kind").getValue() == "all"
                  ? "not_applicable"
                  : "preserve_input",
      shape, "standalone_map_fixture",
      /*derived_from_producer=*/false);
  builder.setInsertionPointAfter(map);
  builder.create<mlir::func::ReturnOp>(location,
                                       map.getResult());
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp>
buildSourceAuthenticatedMap(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source_name =
      "compiler/tests/mlir/map_sin_source_fixture.mdsl";
  const std::string source =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) +
               "/map_sin_source_fixture.mdsl");
  constexpr llvm::StringLiteral expression = "std::sin(value)";
  const std::size_t begin = source.find(expression.str());
  check(begin != std::string::npos,
        "source-authenticated map fixture must contain its sine expression");
  if (begin == std::string::npos)
    return {};
  const std::size_t end = begin + expression.size();
  const std::size_t line_start = source.rfind('\n', begin);
  const unsigned line =
      static_cast<unsigned>(1 + std::count(source.begin(),
                                           source.begin() + begin, '\n'));
  const unsigned column = static_cast<unsigned>(
      begin - (line_start == std::string::npos ? 0 : line_start + 1) + 1);
  const std::array<std::uint8_t, 32> digest =
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(source));
  const std::string snapshot =
      "sha256:" + llvm::toHex(llvm::ArrayRef(digest), true);

  mlir::OpBuilder builder(&context);
  const auto location =
      mlir::FileLineColLoc::get(&context, source_name, line, column);
  auto module = mlir::ModuleOp::create(location);
  const auto tensor =
      mlir::RankedTensorType::get({8, 8}, builder.getF32Type());
  auto function = mlir::func::FuncOp::create(
      location, "source_authenticated_map_fixture",
      builder.getFunctionType({tensor}, {tensor}));
  module.push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  const auto shape = builder.getArrayAttr(
      {scalarStatic(builder, 8), scalarStatic(builder, 8)});
  auto map = builder.create<dialect::MapOp>(
      location, tensor, entry->getArgument(0), mlir::Value{},
      allDomain(builder), builder.getStringAttr("not_applicable"),
      tensorContract(builder, tensor, shape), mapEffects(builder, false),
      mapNumerical(builder),
      sourceAuthenticatedProvenance(
          builder, location, "source_authenticated_map",
          "map_sin_source_fixture", snapshot, begin, end));
  auto *body = new mlir::Block();
  map.getBody().push_back(body);
  mlir::Value element = body->addArgument(builder.getF32Type(), location);
  builder.setInsertionPointToEnd(body);
  auto sin = builder.create<dialect::SinOp>(
      location, builder.getF32Type(), element, sinNumerical(builder),
      sourceAuthenticatedProvenance(
          builder, location, "source_authenticated_expression",
          "map_sin_source_fixture", snapshot, begin, end));
  builder.create<dialect::YieldOp>(location, sin.getResult());
  builder.setInsertionPointAfter(map);
  builder.create<mlir::func::ReturnOp>(location, map.getResult());
  return module;
}

std::pair<unsigned, unsigned> sourceLineColumn(const std::string &source,
                                               std::size_t offset) {
  const std::size_t line_start = source.rfind('\n', offset);
  const unsigned line = static_cast<unsigned>(
      1 + std::count(source.begin(), source.begin() + offset, '\n'));
  const unsigned column = static_cast<unsigned>(
      offset - (line_start == std::string::npos ? 0 : line_start + 1) + 1);
  return {line, column};
}

bridge::BridgeResult buildSourceAuthenticatedComposition(
    v1::Module capture, mlir::MLIRContext &context,
    const std::string &source, llvm::StringRef snapshot) {
  constexpr llvm::StringLiteral source_name =
      "compiler/tests/mlir/map_sin_source_fixture.mdsl";
  constexpr llvm::StringLiteral gemm_expression = "matcore::mdsl::gemm(";
  constexpr llvm::StringLiteral output_expression = "matcore::mdsl::out(C)";
  constexpr llvm::StringLiteral sine_expression =
      "std::sin(c_storage[index])";
  const std::size_t gemm_begin = source.find(gemm_expression.str());
  const std::size_t gemm_close = source.find(");", gemm_begin);
  const std::size_t output_begin = source.find(output_expression.str(),
                                               gemm_begin);
  const std::size_t lhs_begin = source.find(", A,", output_begin);
  const std::size_t rhs_begin = source.find(", B)", output_begin);
  const std::size_t sine_begin = source.find(sine_expression.str());
  check(gemm_begin != std::string::npos && gemm_close != std::string::npos &&
            output_begin != std::string::npos && lhs_begin != std::string::npos &&
            rhs_begin != std::string::npos && sine_begin != std::string::npos,
        "trusted composition fixture must contain exact GEMM and sine expressions");
  if (gemm_begin == std::string::npos || gemm_close == std::string::npos ||
      output_begin == std::string::npos || lhs_begin == std::string::npos ||
      rhs_begin == std::string::npos || sine_begin == std::string::npos)
    return {};

  auto result = buildGemmMap(std::move(capture), context, /*use_slice=*/true);
  if (!result)
    return result;
  const std::size_t gemm_end = gemm_close + 2;
  const std::size_t sine_end = sine_begin + sine_expression.size();
  const auto [gemm_line, gemm_column] =
      sourceLineColumn(source, gemm_begin);
  const auto [sine_line, sine_column] =
      sourceLineColumn(source, sine_begin);
  mlir::OpBuilder builder(&context);
  const auto gemm_location = mlir::FileLineColLoc::get(
      &context, source_name, gemm_line, gemm_column);
  const auto sine_location = mlir::FileLineColLoc::get(
      &context, source_name, sine_line, sine_column);

  mlir::func::FuncOp function;
  dialect::GemmOp gemm;
  dialect::MapOp map;
  dialect::SinOp sin;
  result.module->walk([&](mlir::func::FuncOp operation) { function = operation; });
  result.module->walk([&](dialect::GemmOp operation) { gemm = operation; });
  result.module->walk([&](dialect::MapOp operation) { map = operation; });
  result.module->walk([&](dialect::SinOp operation) { sin = operation; });
  check(function && gemm && map && sin,
        "trusted composition fixture must contain the complete semantic chain");
  if (!function || !gemm || !map || !sin) {
    result.module = nullptr;
    return result;
  }

  map->setAttr("domain", allDomain(builder));
  map.setOutsideDomain("not_applicable");

  (*result.module)->setAttr("mdsl.source_file",
                            builder.getStringAttr(source_name));
  (*result.module)->setAttr("mdsl.translation_unit",
                            builder.getStringAttr(source_name));
  function->setLoc(gemm_location);
  gemm->setLoc(gemm_location);
  mlir::DictionaryAttr provenance = gemm.getProvenance();
  provenance = withField(builder, provenance, "file",
                         builder.getStringAttr(source_name));
  provenance = withField(builder, provenance, "line",
                         builder.getI64IntegerAttr(gemm_line));
  provenance = withField(builder, provenance, "column",
                         builder.getI64IntegerAttr(gemm_column));
  provenance = withField(builder, provenance, "offset",
                         builder.getI64IntegerAttr(gemm_begin));
  provenance = withField(
      builder, provenance, "call_range",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("begin", builder.getI64IntegerAttr(gemm_begin)),
           builder.getNamedAttr("end", builder.getI64IntegerAttr(gemm_end))}));
  const std::size_t lhs_value_begin = lhs_begin + 2;
  const std::size_t rhs_value_begin = rhs_begin + 2;
  provenance = withField(
      builder, provenance, "argument_ranges",
      builder.getArrayAttr(
          {builder.getDictionaryAttr(
               {builder.getNamedAttr(
                    "begin", builder.getI64IntegerAttr(output_begin)),
                builder.getNamedAttr(
                    "end", builder.getI64IntegerAttr(
                               output_begin + output_expression.size()))}),
           builder.getDictionaryAttr(
               {builder.getNamedAttr(
                    "begin", builder.getI64IntegerAttr(lhs_value_begin)),
                builder.getNamedAttr(
                    "end", builder.getI64IntegerAttr(lhs_value_begin + 1))}),
           builder.getDictionaryAttr(
               {builder.getNamedAttr(
                    "begin", builder.getI64IntegerAttr(rhs_value_begin)),
                builder.getNamedAttr(
                    "end", builder.getI64IntegerAttr(rhs_value_begin + 1))})}));
  gemm->setAttr("provenance", provenance);

  map->setLoc(sine_location);
  map->setAttr(
      "provenance",
      sourceAuthenticatedProvenance(
          builder, sine_location, "source_authenticated_map",
          gemm.getSiteId(), snapshot, sine_begin, sine_end));
  sin->setLoc(sine_location);
  sin->setAttr(
      "provenance",
      sourceAuthenticatedProvenance(
          builder, sine_location, "source_authenticated_expression",
          gemm.getSiteId(), snapshot, sine_begin, sine_end));
  return result;
}

dialect::MapOp findMap(mlir::ModuleOp module) {
  dialect::MapOp result;
  module.walk([&](dialect::MapOp operation) { result = operation; });
  return result;
}

mlir::func::FuncOp appendSemanticRootClone(mlir::ModuleOp module,
                                           mlir::OpBuilder &builder,
                                           llvm::StringRef site_id,
                                           std::int64_t capture_ordinal) {
  auto original = *module.getOps<mlir::func::FuncOp>().begin();
  auto duplicate =
      mlir::cast<mlir::func::FuncOp>(original->clone());
  duplicate->setAttr(
      "sym_name",
      builder.getStringAttr("__matcore_semantic_" + site_id.str()));
  duplicate->setAttr("mdsl.site_id", builder.getStringAttr(site_id));
  duplicate->setAttr("mdsl.capture_ordinal",
                     builder.getI64IntegerAttr(capture_ordinal));
  duplicate.walk([&](dialect::GemmOp gemm) {
    gemm->setAttr("site_id", builder.getStringAttr(site_id));
  });
  duplicate.walk([&](dialect::MapOp map) {
    map->setAttr(
        "provenance",
        withField(builder, map.getProvenance(), "source_anchor",
                  builder.getStringAttr(site_id)));
  });
  duplicate.walk([&](dialect::SinOp sin) {
    sin->setAttr(
        "provenance",
        withField(builder, sin.getProvenance(), "source_anchor",
                  builder.getStringAttr(site_id)));
  });
  module.getBody()->push_back(duplicate);
  return duplicate;
}

using MapMutation =
    std::function<void(dialect::MapOp, mlir::OpBuilder &)>;
using ModuleMutation =
    std::function<void(mlir::ModuleOp, mlir::OpBuilder &)>;

void expectRejected(llvm::StringRef valid_text, const MapMutation &mutate,
                    std::string_view message) {
  mlir::MLIRContext context;
  bridge::registerMatcoreSemanticDialects(context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(valid_text, &context);
  check(static_cast<bool>(module), "negative fixture must parse before mutation");
  if (!module)
    return;
  mlir::OpBuilder builder(&context);
  mutate(findMap(*module), builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::failed(mlir::verify(*module)), message);
}

void expectEnvelopeRejected(llvm::StringRef valid_text,
                            const ModuleMutation &mutate,
                            llvm::StringRef expected_reason,
                            std::string_view message) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  bridge::registerMatcoreSemanticDialects(context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(valid_text, &context);
  check(static_cast<bool>(module),
        "composition-envelope negative fixture must parse before mutation");
  if (!module)
    return;
  mlir::OpBuilder builder(&context);
  mutate(*module, builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*module)),
        std::string(message) +
            " (mutation must remain valid generic/dialect MLIR)");
  std::string error;
  check(!dialect::verifyCompositionV1Module(*module, error), message);
  check(error.find(expected_reason.str()) != std::string::npos,
        std::string(message) +
            " (rejection must report the deterministic reason)");
}

void expectAuthenticatedEnvelopeRejected(
    llvm::StringRef valid_text,
    const dialect::AuthenticatedSourceSnapshotV1 &authenticated_source,
    const ModuleMutation &mutate, llvm::StringRef expected_reason,
    std::string_view message) {
  mlir::MLIRContext context;
  bridge::registerMatcoreSemanticDialects(context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(valid_text, &context);
  check(static_cast<bool>(module),
        "trusted-source negative fixture must parse before mutation");
  if (!module)
    return;
  mlir::OpBuilder builder(&context);
  mutate(*module, builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*module)),
        std::string(message) +
            " (mutation must remain structurally valid MLIR)");
  std::string error;
  check(!dialect::verifyCompositionV1Module(
            *module, authenticated_source, error),
        message);
  check(error.find(expected_reason.str()) != std::string::npos,
        std::string(message) +
            " (rejection must report the deterministic reason)");
}

void checkGolden(mlir::ModuleOp module, llvm::StringRef name,
                 std::string &serialized) {
  check(mlir::succeeded(mlir::verify(module)),
        "canonical multi-op module must verify");
  std::string envelope_error;
  check(dialect::verifyCompositionV1Module(module, envelope_error),
        "canonical multi-op module must satisfy composition-v1 envelope");
  serialized = bridge::serializeDeterministicMlir(module);
  const std::string expected =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) + "/" + name.str());
  check(serialized == expected, "canonical module must match golden bytes");
  mlir::MLIRContext parsed_context;
  bridge::registerMatcoreSemanticDialects(parsed_context);
  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(serialized, &parsed_context);
  check(static_cast<bool>(parsed), "canonical text must parse and verify");
  check(parsed && bridge::serializeDeterministicMlir(*parsed) == serialized,
        "canonical text must be byte-stable across parse/print");
  check(parsed &&
            dialect::verifyCompositionV1Module(*parsed, envelope_error),
        "reparsed canonical text must satisfy composition-v1 envelope");
}

void checkIndependentSlicePreservation() {
  std::array<float, 64> input{};
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(index) + 0.25F;
  input[0] = std::bit_cast<float>(std::uint32_t{0x80000000});
  input[7] = std::bit_cast<float>(std::uint32_t{0x7fc12345});
  std::array<float, 64> output = input;
  for (std::size_t row = 1; row < 7; row += 2)
    for (std::size_t column = 2; column < 6; ++column)
      output[row * 8 + column] = std::sin(input[row * 8 + column]);

  std::size_t preserved = 0;
  for (std::size_t row = 0; row < 8; ++row) {
    for (std::size_t column = 0; column < 8; ++column) {
      const bool active = row >= 1 && row < 7 && ((row - 1) % 2 == 0) &&
                          column >= 2 && column < 6;
      if (active)
        continue;
      ++preserved;
      check(std::bit_cast<std::uint32_t>(output[row * 8 + column]) ==
                std::bit_cast<std::uint32_t>(input[row * 8 + column]),
            "partial slice evaluator must byte-preserve every inactive element");
    }
  }
  check(preserved == 52,
        "partial slice evaluator must preserve the exact inactive cardinality");
}

} // namespace

int main() {
  const v1::Module capture = readCapture();

  mlir::MLIRContext all_context;
  auto all = buildGemmMap(capture, all_context, /*use_slice=*/false);
  check(static_cast<bool>(all), "dynamic GEMM-to-SIN(all) fixture must build");
  std::string all_text;
  if (all)
    checkGolden(*all.module, "gemm_sin_all.semantic.golden.mlir", all_text);

  mlir::MLIRContext slice_context;
  auto slice = buildGemmMap(capture, slice_context, /*use_slice=*/true);
  check(static_cast<bool>(slice), "static partial-slice fixture must build");
  std::string slice_text;
  if (slice)
    checkGolden(*slice.module, "gemm_sin_slice.semantic.golden.mlir",
                slice_text);

  if (all) {
    dialect::GemmOp gemm;
    all.module->walk([&](dialect::GemmOp operation) { gemm = operation; });
    dialect::MapOp map = findMap(*all.module);
    check(map.getInput() == gemm.getResult(),
          "SSA use-def must encode GEMM-to-map dependency");
    check(mlir::isMemoryEffectFree(map.getOperation()),
          "functional map must be effect-free");
    check(!mlir::isMemoryEffectFree(gemm.getOperation()) &&
              !mlir::wouldOpBeTriviallyDead(gemm.getOperation()),
          "GEMM destination write must remain observable before functional map");
    check(map.getInput().getType() == map.getResult().getType(),
          "map must preserve the exact tensor type");
    check(map.getOutsideDomain() == "not_applicable" && !map.getMask(),
          "all domain must be canonical and mask-free");
  }

  mlir::MLIRContext domains_context;
  bridge::registerMatcoreSemanticDialects(domains_context);
  mlir::Builder domains_builder(&domains_context);
  auto indices = buildStandaloneMap(domains_context,
                                    indicesDomain(domains_builder), false);
  auto mask =
      buildStandaloneMap(domains_context, maskDomain(domains_builder), true);
  auto dynamic_mask = buildStandaloneMap(
      domains_context, maskDomain(domains_builder), true,
      /*dynamic_shape=*/true);
  auto incompatible_mask = buildStandaloneMap(
      domains_context, maskDomain(domains_builder), true,
      /*dynamic_shape=*/false, /*rank_three=*/false,
      /*mask_columns=*/7);
  auto rank_three = buildStandaloneMap(
      domains_context, allDomain(domains_builder), false,
      /*dynamic_shape=*/false, /*rank_three=*/true);
  auto encoded_tensor = buildStandaloneMap(
      domains_context, allDomain(domains_builder), false,
      /*dynamic_shape=*/false, /*rank_three=*/false,
      /*mask_columns=*/8, /*encoded_tensor=*/true);
  auto encoded_mask = buildStandaloneMap(
      domains_context, maskDomain(domains_builder), true,
      /*dynamic_shape=*/false, /*rank_three=*/false,
      /*mask_columns=*/8, /*encoded_tensor=*/false,
      /*encoded_mask=*/true);
  auto source_authenticated = buildSourceAuthenticatedMap(domains_context);
  const std::string authenticated_source_bytes =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) +
               "/map_sin_source_fixture.mdsl");
  const std::array<std::uint8_t, 32> authenticated_source_hash =
      llvm::SHA256::hash(
          llvm::arrayRefFromStringRef(authenticated_source_bytes));
  const std::string authenticated_source_digest =
      "sha256:" +
      llvm::toHex(llvm::ArrayRef(authenticated_source_hash), true);
  mlir::MLIRContext authenticated_composition_context;
  auto authenticated_composition = buildSourceAuthenticatedComposition(
      capture, authenticated_composition_context, authenticated_source_bytes,
      authenticated_source_digest);
  const dialect::AuthenticatedSourceSnapshotV1 authenticated_source_context{
      "compiler/tests/mlir/map_sin_source_fixture.mdsl",
      authenticated_source_digest, authenticated_source_bytes,
      authenticated_source_bytes.size()};
  check(mlir::succeeded(mlir::verify(*indices)),
        "closed indices domain must verify");
  check(mlir::succeeded(mlir::verify(*mask)),
        "closed predicate mask domain must verify");
  check(mlir::succeeded(mlir::verify(*dynamic_mask)),
        "dynamic mask shapes must remain legal behind an explicit guard obligation");
  check(source_authenticated &&
            mlir::succeeded(mlir::verify(*source_authenticated)),
        "source-authenticated syntax must remain structurally verifiable against a real textual fixture");
  check(authenticated_composition &&
            mlir::succeeded(mlir::verify(*authenticated_composition.module)),
        "source-authenticated composition must be structurally valid before trust verification");
  if (authenticated_composition) {
    std::string authenticated_error;
    check(!dialect::verifyCompositionV1Module(
              *authenticated_composition.module, authenticated_error) &&
              authenticated_error.find("trusted source snapshot context") !=
                  std::string::npos,
          "context-free production composition verification must fail closed on source-authenticated provenance");
    check(dialect::verifyCompositionV1Module(
              *authenticated_composition.module, authenticated_source_context,
              authenticated_error),
          "trusted composition verification must authenticate every source-backed operation");
  }
  check(findMap(*dynamic_mask)
            .getDomain()
            .getAs<mlir::StringAttr>("shape_equality")
            .getValue() == "required_precondition",
        "dynamic question-mark dimensions must not be treated as proven equal");
  {
    mlir::ScopedDiagnosticHandler silence(
        &domains_context, [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::failed(mlir::verify(*incompatible_mask)),
          "statically incompatible mask shapes must be rejected");
    check(mlir::failed(mlir::verify(*rank_three)),
          "version-1 map must fail closed for rank greater than two");
    check(mlir::failed(mlir::verify(*encoded_tensor)),
          "host row-major map tensors must reject non-null encodings");
    check(mlir::failed(mlir::verify(*encoded_mask)),
          "host predicate masks must reject non-null encodings");
  }
  checkIndependentSlicePreservation();

  if (all) {
    dialect::GemmOp gemm;
    all.module->walk([&](dialect::GemmOp operation) { gemm = operation; });
    const auto result_type =
        mlir::cast<mlir::RankedTensorType>(gemm.getResult().getType());
    gemm.getResult().setType(mlir::RankedTensorType::get(
        result_type.getShape(), result_type.getElementType(),
        mlir::StringAttr::get(&all_context, "test.encoding")));
    mlir::ScopedDiagnosticHandler silence(
        &all_context, [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::failed(mlir::verify(*all.module)),
          "GEMM host row-major values must reject non-null tensor encodings");
  }

  check(all_text.find("kind = \"source_expression\"") == std::string::npos,
        "derived GEMM-to-SIN fixture must never forge source-expression provenance");

  if (!all_text.empty()) {
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr("domain", withField(builder, map.getDomain(), "kind",
                                            builder.getStringAttr("other")));
        },
        "unknown domain kind must fail closed");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr("domain", withField(builder, map.getDomain(), "version",
                                            builder.getI32IntegerAttr(2)));
        },
        "unknown domain version must fail closed");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          llvm::SmallVector<mlir::NamedAttribute> attributes(
              map.getDomain().begin(), map.getDomain().end());
          attributes.push_back(
              builder.getNamedAttr("extra", builder.getStringAttr("bad")));
          map->setAttr("domain", builder.getDictionaryAttr(attributes));
        },
        "domain dictionaries must reject extra fields");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &) {
          map.setOutsideDomain("preserve_input");
        },
        "all domain must reject inactive-element semantics");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr("domain", maskDomain(builder));
          map.setOutsideDomain("preserve_input");
        },
        "mask domain without predicate operand must be rejected");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "tensor_contract",
              withField(builder, map.getTensorContract(), "aliasing",
                        builder.getStringAttr("inplace")));
        },
        "in-place tensor aliasing must be rejected");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "tensor_contract",
              withField(builder, map.getTensorContract(), "result_mutability",
                        builder.getStringAttr("read_write")));
        },
        "functional result mutability must be exact");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "tensor_contract",
              withoutField(builder, map.getTensorContract(), "shape"));
        },
        "tensor contract must reject missing shape");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "tensor_contract",
              withField(builder, map.getTensorContract(), "shape",
                        builder.getArrayAttr({scalarDynamic(builder, "x"),
                                              scalarDynamic(builder, "y")})));
        },
        "dynamic tensor contract symbols must propagate from the GEMM result rather than be guessed");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "effects",
              withField(builder, map.getEffects(), "writes",
                        builder.getArrayAttr({builder.getStringAttr("result")})));
        },
        "functional map must reject write effects");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "numerical",
              withField(builder, map.getNumerical(), "inplace",
                        builder.getBoolAttr(true)));
        },
        "map numerical contract must reject in-place execution");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "numerical",
              withField(builder, map.getNumerical(), "approximate_math",
                        builder.getBoolAttr(true)));
        },
        "map numerical contract must reject approximation");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "file",
                        builder.getStringAttr("other.mdsl")));
        },
        "map provenance must match its source location");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr("provenance",
                       withoutField(builder, map.getProvenance(),
                                    "source_anchor"));
        },
        "map provenance must retain a nonempty source anchor");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "source_anchor",
                        builder.getStringAttr("different_site")));
        },
        "map provenance must authenticate the producing GEMM site");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &) {
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin->setLoc(mlir::UnknownLoc::get(map.getContext()));
        },
        "scalar source provenance must reject a missing FileLineCol location");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin->setAttr(
              "provenance",
              withField(builder, sin.getProvenance(), "source_anchor",
                        builder.getStringAttr("different_expression")));
        },
        "scalar provenance must remain tied to the map source anchor");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map.getResult().setType(
              mlir::RankedTensorType::get({1, 1}, builder.getF32Type()));
        },
        "functional result type must exactly preserve the input tensor type");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin->setAttr("numerical",
                       withoutField(builder, sin.getNumerical(), "accuracy"));
        },
        "sin numerical contract must require finite accuracy");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin->setAttr(
              "numerical",
              withField(builder, sin.getNumerical(), "infinity",
                        builder.getStringAttr("assume_absent")));
        },
        "sin numerical contract must preserve explicit infinity behavior");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin->setAttr(
              "numerical",
              withField(builder, sin.getNumerical(), "signed_zero",
                        builder.getStringAttr("relaxed")));
        },
        "sin numerical contract must preserve signed zero");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &) {
          mlir::Block &block = map.getBody().front();
          auto sin = mlir::cast<dialect::SinOp>(&block.front());
          auto yield = mlir::cast<dialect::YieldOp>(block.getTerminator());
          yield.setOperand(sin.getInput());
          sin.erase();
        },
        "identity/empty map region must be rejected in version 1");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &builder) {
          mlir::Block &block = map.getBody().front();
          auto first = mlir::cast<dialect::SinOp>(&block.front());
          builder.setInsertionPointAfter(first);
          auto second = builder.create<dialect::SinOp>(
              first.getLoc(), builder.getF32Type(), first.getInput(),
              first.getNumerical(), first.getProvenance());
          auto yield = mlir::cast<dialect::YieldOp>(block.getTerminator());
          yield.setOperand(second.getResult());
        },
        "nonlinear/dead scalar region chains must be rejected");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &) {
          map.getBody().getBlocks().clear();
        },
        "map must reject a zero-block body");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &builder) {
          auto *extra = new mlir::Block();
          map.getBody().push_back(extra);
          auto element = extra->addArgument(builder.getF32Type(), map.getLoc());
          builder.setInsertionPointToEnd(extra);
          auto sin = builder.create<dialect::SinOp>(
              map.getLoc(), builder.getF32Type(), element,
              mlir::cast<dialect::SinOp>(&map.getBody().front().front())
                  .getNumerical(),
              mlir::cast<dialect::SinOp>(&map.getBody().front().front())
                  .getProvenance());
          builder.create<dialect::YieldOp>(map.getLoc(), sin.getResult());
        },
        "map must reject a two-block body");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map.getBody().front().getArgument(0).setType(builder.getI32Type());
        },
        "map must reject a wrong scalar block-argument type");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map.getBody().front().addArgument(builder.getF32Type(), map.getLoc());
        },
        "map must reject more than one block argument");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::Builder &) {
          map.getBody().front().getTerminator()->erase();
        },
        "map must reject a missing terminator");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &builder) {
          map.getBody().front().getTerminator()->erase();
          builder.setInsertionPointToEnd(&map.getBody().front());
          builder.create<mlir::func::ReturnOp>(map.getLoc());
        },
        "map must reject the wrong terminator operation");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &builder) {
          auto yield = mlir::cast<dialect::YieldOp>(
              map.getBody().front().getTerminator());
          builder.setInsertionPointAfter(yield);
          builder.create<dialect::YieldOp>(yield.getLoc(), yield.getValue());
        },
        "map must reject multiple terminators");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &builder) {
          auto function = map->getParentOfType<mlir::func::FuncOp>();
          llvm::SmallVector<mlir::Type> inputs(
              function.getFunctionType().getInputs());
          inputs.push_back(builder.getF32Type());
          function.setFunctionType(builder.getFunctionType(
              inputs, function.getFunctionType().getResults()));
          auto captured = function.getBody().front().addArgument(
              builder.getF32Type(), map.getLoc());
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin.setOperand(captured);
        },
        "isolated map region must reject values captured from the parent function");
    expectRejected(
        all_text,
        [](dialect::MapOp map, mlir::OpBuilder &builder) {
          mlir::Block &body = map.getBody().front();
          builder.setInsertionPoint(&body.front());
          builder.create<mlir::func::CallOp>(
              map.getLoc(), "opaque_side_effect", mlir::TypeRange{},
              mlir::ValueRange{});
        },
        "map body allowlist must reject non-Matcore/side-effecting operations");

    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          module->removeAttr("mdsl.composition_schema");
        },
        "exact schema/version",
        "composition envelope must reject a missing schema marker");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          module->setAttr("mdsl.composition_version",
                          builder.getI32IntegerAttr(2));
        },
        "exact schema/version",
        "composition envelope must reject an unsupported version");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          module->setAttr("mdsl.source_file",
                          builder.getStringAttr("different.mdsl"));
        },
        "function location must name the module source",
        "composition envelope must bind functions to the declared module source");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          module.walk([](mlir::func::FuncOp function) {
            function->setLoc(mlir::FileLineColLoc::get(
                function.getContext(), "different.mdsl", 21, 3));
          });
        },
        "function location must name the module source",
        "composition envelope must reject a foreign semantic-root location");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          module.walk([&](mlir::func::FuncOp function) {
            constexpr llvm::StringLiteral changed_site =
                "mc_00000000000000000000000000000000";
            function->setAttr("mdsl.site_id",
                              builder.getStringAttr(changed_site));
            function->setAttr(
                "sym_name",
                builder.getStringAttr("__matcore_semantic_" +
                                      changed_site.str()));
          });
        },
        "GEMM site must match",
        "composition envelope must tie the semantic root to its GEMM site");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          auto function = *module.getOps<mlir::func::FuncOp>().begin();
          mlir::Operation *duplicate = function->clone();
          duplicate->setAttr(
              "sym_name",
              builder.getStringAttr("__matcore_semantic_duplicate_root"));
          module.getBody()->push_back(duplicate);
        },
        "duplicate semantic site IDs",
        "composition envelope must reject duplicate semantic-root site identities");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          auto function = *module.getOps<mlir::func::FuncOp>().begin();
          function->setAttr("sym_name",
                            builder.getStringAttr("semantic_root_alias"));
        },
        "function symbol must match",
        "composition envelope must bind each function symbol to its site identity");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          auto function = *module.getOps<mlir::func::FuncOp>().begin();
          function->setAttr("mdsl.capture_ordinal",
                            builder.getI64IntegerAttr(1));
        },
        "capture ordinals must be contiguous",
        "composition envelope must require a zero-based first capture ordinal");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          appendSemanticRootClone(
              module, builder, "mc_11111111111111111111111111111111", 0);
        },
        "capture ordinals must be contiguous",
        "composition envelope must reject duplicate capture ordinals");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          mlir::func::FuncOp second = appendSemanticRootClone(
              module, builder, "mc_11111111111111111111111111111111", 1);
          second->moveBefore(&module.getBody()->front());
        },
        "capture ordinals must be contiguous",
        "composition envelope must reject semantic-root reordering");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          dialect::GemmOp gemm;
          module.walk([&](dialect::GemmOp operation) { gemm = operation; });
          const mlir::Value lhs = gemm.getLhs();
          gemm.getLhsMutable().assign(gemm.getRhs());
          gemm.getRhsMutable().assign(lhs);
        },
        "preserve lhs/rhs/output root argument order",
        "composition envelope must reject root operand permutation");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::MapOp map = findMap(module);
          map->setLoc(mlir::FileLineColLoc::get(
              map.getContext(), "compiler/tests/frontend/gemm_capture.mdsl",
              22, 3));
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "line",
                        builder.getI64IntegerAttr(22)));
        },
        "derived map location must equal",
        "composition envelope must tie a derived map location to its producer");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::MapOp map = findMap(module);
          auto sin = mlir::cast<dialect::SinOp>(&map.getBody().front().front());
          sin->setLoc(mlir::FileLineColLoc::get(
              sin.getContext(), "compiler/tests/frontend/gemm_capture.mdsl",
              22, 3));
          sin->setAttr(
              "provenance",
              withField(builder, sin.getProvenance(), "line",
                        builder.getI64IntegerAttr(22)));
        },
        "derived scalar location must equal",
        "composition envelope must tie a derived scalar location to its map");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::MapOp map = findMap(module);
          auto provenance = withField(
              builder, map.getProvenance(), "authenticity",
              builder.getStringAttr("synthetic_test_fixture"));
          map->setAttr(
              "provenance",
              withField(builder, provenance, "kind",
                        builder.getStringAttr("synthetic_semantic_composition")));
        },
        "must be non-synthetic",
        "composition envelope must reject synthetic fixture provenance");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          auto helper = mlir::func::FuncOp::create(
              module.getLoc(), "unsupported_helper",
              builder.getFunctionType({}, {}));
          module.push_back(helper);
          mlir::Block *entry = helper.addEntryBlock();
          builder.setInsertionPointToEnd(entry);
          builder.create<mlir::func::ReturnOp>(helper.getLoc());
        },
        "three tensor arguments",
        "composition envelope must reject unsupported helper functions");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          mlir::OperationState state(module.getLoc(), "review.top_level");
          module.getBody()->push_back(mlir::Operation::create(state));
        },
        "only func.func at module scope",
        "composition envelope must reject unknown top-level operations");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          mlir::func::FuncOp function;
          module.walk([&](mlir::func::FuncOp candidate) { function = candidate; });
          auto return_op = mlir::cast<mlir::func::ReturnOp>(
              function.getBody().front().getTerminator());
          builder.setInsertionPoint(return_op);
          builder.create<mlir::func::CallOp>(
              function.getLoc(), function.getSymName(),
              function.getFunctionType().getResults(),
              function.getBody().front().getArguments());
        },
        "forbids function calls",
        "composition envelope must reject calls even when generic MLIR accepts them");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          mlir::func::FuncOp function;
          module.walk([&](mlir::func::FuncOp candidate) { function = candidate; });
          builder.setInsertionPoint(function.getBody().front().getTerminator());
          mlir::OperationState state(function.getLoc(), "review.effectful");
          state.addAttribute("effect", builder.getStringAttr("unknown"));
          builder.insert(mlir::Operation::create(state));
        },
        "unsupported operation",
        "composition envelope must reject unknown-effect function operations");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::GemmOp gemm;
          module.walk([&](dialect::GemmOp operation) { gemm = operation; });
          builder.setInsertionPointAfter(gemm);
          builder.clone(*gemm.getOperation());
        },
        "exactly one GEMM",
        "composition envelope must reject multiple GEMM semantic roots in one function");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          dialect::GemmOp gemm;
          mlir::func::ReturnOp return_op;
          module.walk([&](dialect::GemmOp operation) { gemm = operation; });
          module.walk(
              [&](mlir::func::ReturnOp operation) { return_op = operation; });
          return_op.setOperand(0, gemm.getResult());
        },
        "unused functional map",
        "composition envelope must reject dead semantic map results");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          dialect::GemmOp gemm;
          mlir::func::ReturnOp return_op;
          module.walk([&](dialect::GemmOp operation) { gemm = operation; });
          module.walk(
              [&](mlir::func::ReturnOp operation) { return_op = operation; });
          dialect::MapOp map = findMap(module);
          return_op.setOperand(0, gemm.getResult());
          map.erase();
        },
        "at least one map",
        "composition envelope must reject a falsely labeled GEMM-only module");
    expectEnvelopeRejected(
        all_text,
        [](mlir::ModuleOp module, mlir::OpBuilder &) {
          dialect::MapOp map = findMap(module);
          auto function = map->getParentOfType<mlir::func::FuncOp>();
          auto return_op = mlir::cast<mlir::func::ReturnOp>(
              function.getBody().front().getTerminator());
          return_op.setOperand(0, function.getBody().front().getArgument(2));
          map.erase();
        },
        "return must expose",
        "composition envelope must reject a return that bypasses semantic results");
  }

  if (!slice_text.empty()) {
    expectRejected(
        slice_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto domain = map.getDomain();
          domain = withField(
              builder, domain, "begin",
              builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                    builder.getI64IntegerAttr(0)}));
          domain = withField(
              builder, domain, "end",
              builder.getArrayAttr({builder.getI64IntegerAttr(8),
                                    builder.getI64IntegerAttr(8)}));
          domain = withField(
              builder, domain, "step",
              builder.getArrayAttr({builder.getI64IntegerAttr(1),
                                    builder.getI64IntegerAttr(1)}));
          map->setAttr("domain", domain);
        },
        "full slice must canonicalize to all rather than remain ambiguous");
    expectRejected(
        slice_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "domain",
              withField(builder, map.getDomain(), "end",
                        builder.getArrayAttr({builder.getI64IntegerAttr(9),
                                              builder.getI64IntegerAttr(6)})));
        },
        "slice bounds must reject out-of-range coordinates");
    expectRejected(
        slice_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "domain",
              withField(builder, map.getDomain(), "step",
                        builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                              builder.getI64IntegerAttr(1)})));
        },
        "slice steps must be positive");
  }

  const std::string indices_text =
      bridge::serializeDeterministicMlir(*indices);
  {
    mlir::MLIRContext reparsed_context;
    bridge::registerMatcoreSemanticDialects(reparsed_context);
    auto reparsed = mlir::parseSourceString<mlir::ModuleOp>(
        indices_text, &reparsed_context);
    check(reparsed &&
              bridge::serializeDeterministicMlir(*reparsed) == indices_text,
          "canonical lexicographic indices must have byte-stable serialization");
  }
  expectRejected(
      indices_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr(
            "domain",
            withField(
                builder, map.getDomain(), "coordinates",
                builder.getArrayAttr(
                    {builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                           builder.getI64IntegerAttr(1)}),
                     builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                           builder.getI64IntegerAttr(1)})})));
      },
      "indices domain must reject duplicate coordinates");
  expectRejected(
      indices_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr(
            "domain",
            withField(
                builder, map.getDomain(), "coordinates",
                builder.getArrayAttr(
                    {builder.getArrayAttr({builder.getI64IntegerAttr(8),
                                           builder.getI64IntegerAttr(1)})})));
      },
      "indices domain must reject out-of-range coordinates");
  expectRejected(
      indices_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr(
            "domain",
            withField(
                builder, map.getDomain(), "coordinates",
                builder.getArrayAttr(
                    {builder.getArrayAttr({builder.getI64IntegerAttr(7),
                                           builder.getI64IntegerAttr(6)}),
                     builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                           builder.getI64IntegerAttr(1)})})));
      },
      "indices domain must reject reversed coordinate order");
  expectRejected(
      indices_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr(
            "domain",
            withField(
                builder, map.getDomain(), "coordinates",
                builder.getArrayAttr(
                    {builder.getArrayAttr({builder.getI64IntegerAttr(0),
                                           builder.getI64IntegerAttr(1)}),
                     builder.getArrayAttr({builder.getI64IntegerAttr(1),
                                           builder.getI64IntegerAttr(2)}),
                     builder.getArrayAttr({builder.getI64IntegerAttr(1),
                                           builder.getI64IntegerAttr(0)})})));
      },
      "indices domain must reject noncanonical coordinate permutations");
  expectRejected(
      indices_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        llvm::SmallVector<mlir::Attribute> coordinates;
        for (std::int64_t row = 0; row < 8; ++row) {
          for (std::int64_t column = 0; column < 8; ++column) {
            coordinates.push_back(builder.getArrayAttr(
                {builder.getI64IntegerAttr(row),
                 builder.getI64IntegerAttr(column)}));
          }
        }
        map->setAttr(
            "domain",
            withField(builder, map.getDomain(), "coordinates",
                      builder.getArrayAttr(coordinates)));
      },
      "a statically complete indices set must canonicalize to domain(all)");

  const std::string mask_text = bridge::serializeDeterministicMlir(*mask);
  expectRejected(
      mask_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr("domain", allDomain(builder));
        map.setOutsideDomain("not_applicable");
      },
      "non-mask domains must reject a predicate operand");
  expectRejected(
      mask_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr("domain",
                     withoutField(builder, map.getDomain(),
                                  "shape_equality"));
      },
      "mask domain must encode shape equality as a required precondition");
  expectRejected(
      mask_text,
      [](dialect::MapOp map, mlir::Builder &builder) {
        map->setAttr(
            "effects",
            withField(builder, map.getEffects(), "reads",
                      builder.getArrayAttr({builder.getStringAttr("input")})));
      },
      "mask map effects must include the predicate read");

  if (source_authenticated) {
    const std::string source_text =
        bridge::serializeDeterministicMlir(*source_authenticated);
    check(source_text.find("source_authenticated_expression") !=
              std::string::npos &&
              source_text.find("source_snapshot = \"sha256:") !=
                  std::string::npos,
          "source-authenticated fixture must serialize its exact provenance discriminant and snapshot identity");
    expectRejected(
        source_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "provenance",
              withoutField(builder, map.getProvenance(), "source_range"));
        },
        "source-authenticated provenance must require an exact source range");
    expectRejected(
        source_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "source_snapshot",
                        builder.getStringAttr("sha256:not-a-digest")));
        },
        "source-authenticated provenance must reject malformed snapshot identities");
    expectRejected(
        source_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto range = map.getProvenance().getAs<mlir::DictionaryAttr>(
              "source_range");
          range = withAddedField(builder, range, "extra",
                                 builder.getI64IntegerAttr(1));
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "source_range", range));
        },
        "source-authenticated source ranges must reject unknown fields");
    expectRejected(
        source_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "authenticity",
                        builder.getStringAttr("derived_from_producer")));
        },
        "provenance authenticity and kind must form an exact cross-product");
    expectRejected(
        source_text,
        [](dialect::MapOp map, mlir::Builder &builder) {
          auto sin = mlir::cast<dialect::SinOp>(
              &map.getBody().front().front());
          sin->setAttr(
              "provenance",
              withField(builder, sin.getProvenance(), "kind",
                        builder.getStringAttr("derived_elementwise_expression")));
        },
        "source-authenticated scalar provenance must reject a derived kind");
  }

  if (authenticated_composition) {
    const std::string authenticated_text =
        bridge::serializeDeterministicMlir(
            *authenticated_composition.module);
    mlir::MLIRContext reparsed_context;
    bridge::registerMatcoreSemanticDialects(reparsed_context);
    auto reparsed = mlir::parseSourceString<mlir::ModuleOp>(
        authenticated_text, &reparsed_context);
    std::string authenticated_error;
    check(reparsed &&
              dialect::verifyCompositionV1Module(
                  *reparsed, authenticated_source_context,
                  authenticated_error) &&
              bridge::serializeDeterministicMlir(*reparsed) ==
                  authenticated_text,
          "trusted source-authenticated composition must remain byte-stable and authenticated after parse/print");

    const dialect::AuthenticatedSourceSnapshotV1 wrong_identity{
        "different.mdsl", authenticated_source_digest,
        authenticated_source_bytes, authenticated_source_bytes.size()};
    check(!dialect::verifyCompositionV1Module(
              *authenticated_composition.module, wrong_identity,
              authenticated_error) &&
              authenticated_error.find("identity") != std::string::npos,
          "trusted context must bind the exact module source identity");

    const std::string wrong_digest =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    const dialect::AuthenticatedSourceSnapshotV1 wrong_digest_context{
        authenticated_source_context.source_identity, wrong_digest,
        authenticated_source_bytes, authenticated_source_bytes.size()};
    check(!dialect::verifyCompositionV1Module(
              *authenticated_composition.module, wrong_digest_context,
              authenticated_error) &&
              authenticated_error.find("digest") != std::string::npos,
          "trusted context digest must be recomputed from the supplied bytes");

    const dialect::AuthenticatedSourceSnapshotV1 wrong_length{
        authenticated_source_context.source_identity,
        authenticated_source_digest, authenticated_source_bytes,
        authenticated_source_bytes.size() + 1};
    check(!dialect::verifyCompositionV1Module(
              *authenticated_composition.module, wrong_length,
              authenticated_error) &&
              authenticated_error.find("byte length") != std::string::npos,
          "trusted context must bind the exact source byte length");

    expectAuthenticatedEnvelopeRejected(
        authenticated_text, authenticated_source_context,
        [&](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::MapOp map = findMap(module);
          auto range = map.getProvenance().getAs<mlir::DictionaryAttr>(
              "source_range");
          range = withField(
              builder, range, "end",
              builder.getI64IntegerAttr(authenticated_source_bytes.size() +
                                        1));
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "source_range", range));
        },
        "byte bounds",
        "trusted source verification must reject a syntactically valid range beyond the authenticated bytes");
    expectAuthenticatedEnvelopeRejected(
        authenticated_text, authenticated_source_context,
        [](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::MapOp map = findMap(module);
          const auto provenance_line =
              map.getProvenance().getAs<mlir::IntegerAttr>("line");
          const auto provenance_column =
              map.getProvenance().getAs<mlir::IntegerAttr>("column");
          const auto file =
              map.getProvenance().getAs<mlir::StringAttr>("file");
          const std::int64_t changed_line = provenance_line.getInt() + 1;
          map->setLoc(mlir::FileLineColLoc::get(
              map.getContext(), file.getValue(),
              static_cast<unsigned>(changed_line),
              static_cast<unsigned>(provenance_column.getInt())));
          map->setAttr(
              "provenance",
              withField(builder, map.getProvenance(), "line",
                        builder.getI64IntegerAttr(changed_line)));
        },
        "line/column",
        "trusted source verification must derive line/column from the authenticated range begin");
    expectAuthenticatedEnvelopeRejected(
        authenticated_text, authenticated_source_context,
        [&](mlir::ModuleOp module, mlir::OpBuilder &builder) {
          dialect::SinOp sin;
          module.walk([&](dialect::SinOp operation) { sin = operation; });
          sin->setAttr(
              "provenance",
              withField(builder, sin.getProvenance(), "source_snapshot",
                        builder.getStringAttr(wrong_digest)));
        },
        "identity, digest, or byte bounds",
        "trusted source verification must authenticate every nested source-backed scalar operation");
  }

  std::cout << "Matcore MLIR map/domain: " << checks << " checks, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
