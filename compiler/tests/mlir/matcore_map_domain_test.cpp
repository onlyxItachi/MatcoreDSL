#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"

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
                                        llvm::StringRef source_anchor) {
  const auto source = mlir::cast<mlir::FileLineColLoc>(location);
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("column",
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
                            llvm::StringRef source_anchor) {
  const auto type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto map = builder.create<dialect::MapOp>(
      location, type, input, mask, domain,
      builder.getStringAttr(outside_domain), tensorContract(builder, type, shape),
      mapEffects(builder, static_cast<bool>(mask)), mapNumerical(builder),
      semanticProvenance(builder, location, "semantic_composition",
                         source_anchor));
  auto *body = new mlir::Block();
  map.getBody().push_back(body);
  mlir::Value element = body->addArgument(builder.getF32Type(), location);
  builder.setInsertionPointToEnd(body);
  auto sin = builder.create<dialect::SinOp>(
      location, builder.getF32Type(), element, sinNumerical(builder),
      semanticProvenance(builder, location, "source_expression",
                         source_anchor));
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
                          shape, gemm.getSiteId());
  return_op.setOperand(0, map.getResult());
  return result;
}

mlir::OwningOpRef<mlir::ModuleOp>
buildStandaloneMap(mlir::MLIRContext &context, mlir::DictionaryAttr domain,
                   bool with_mask, bool dynamic_shape = false,
                   bool rank_three = false, int64_t mask_columns = 8) {
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
      builder.getF32Type());
  const auto mask = mlir::RankedTensorType::get(
      dynamic_shape
          ? llvm::ArrayRef<int64_t>{mlir::ShapedType::kDynamic,
                                    mlir::ShapedType::kDynamic}
          : llvm::ArrayRef<int64_t>{8, mask_columns},
      builder.getI1Type());
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
      shape, "standalone_map_fixture");
  builder.setInsertionPointAfter(map);
  builder.create<mlir::func::ReturnOp>(location,
                                       map.getResult());
  return module;
}

dialect::MapOp findMap(mlir::ModuleOp module) {
  dialect::MapOp result;
  module.walk([&](dialect::MapOp operation) { result = operation; });
  return result;
}

using MapMutation =
    std::function<void(dialect::MapOp, mlir::OpBuilder &)>;

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

void checkGolden(mlir::ModuleOp module, llvm::StringRef name,
                 std::string &serialized) {
  check(mlir::succeeded(mlir::verify(module)),
        "canonical multi-op module must verify");
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
  check(mlir::succeeded(mlir::verify(*indices)),
        "closed indices domain must verify");
  check(mlir::succeeded(mlir::verify(*mask)),
        "closed predicate mask domain must verify");
  check(mlir::succeeded(mlir::verify(*dynamic_mask)),
        "dynamic mask shapes must remain legal behind an explicit guard obligation");
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
  }
  checkIndependentSlicePreservation();

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

  std::cout << "Matcore MLIR map/domain: " << checks << " checks, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
