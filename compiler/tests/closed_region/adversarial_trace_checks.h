#ifndef MDSLC_CLOSED_REGION_ADVERSARIAL_TRACE_CHECKS_H
#define MDSLC_CLOSED_REGION_ADVERSARIAL_TRACE_CHECKS_H

#include "ClosedRegionAdmission.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace matcore::mdslc::test {

// Independent structural oracle for the rhs_rectangular_late_read source.
// This follows SSA sequencing ancestry; it does not evaluate C++, GEMM,
// predicates, providers, storage writes, or a failure simulator. A failure at
// check1 preserves this *required semantic prefix*, conditional on the prefix's
// own effects succeeding. It proves no physical publication or rollback.
inline void checkClosedRegionAdversarialTrace(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    mlir::ModuleOp module,
    const std::function<void(bool, const std::string &)> &check) {
  std::vector<mlir::Operation *> reads, guards, gemms, publications, observations;
  module.walk([&](mlir::Operation *op) {
    const auto name = op->getName().getStringRef();
    if (name == "mdsl_admission.read") reads.push_back(op);
    if (name == "mdsl_admission.check_gemm") guards.push_back(op);
    if (name == "mdsl_admission.gemm") gemms.push_back(op);
    if (name == "mdsl_admission.publish") publications.push_back(op);
    if (name == "mdsl_admission.observe") observations.push_back(op);
  });
  const bool specimen = reads.size() == 3 && guards.size() == 2 &&
                        gemms.size() == 2 && publications.size() == 2 &&
                        observations.size() == 1;
  check(specimen, "independent trace oracle has the exact RHS specimen");
  if (!specimen) return;

  std::vector<mlir::Operation *> ancestors;
  auto token = guards[1]->getOperand(0);
  bool reached_entry = false;
  for (unsigned depth = 0; depth != 64; ++depth) {
    auto *producer = token.getDefiningOp();
    if (!producer) break;
    if (producer->getName().getStringRef() == "mdsl_admission.begin") {
      reached_entry = true;
      break;
    }
    ancestors.push_back(producer);
    if (!producer->getNumOperands()) break;
    token = producer->getOperand(0);
  }
  std::reverse(ancestors.begin(), ancestors.end());
  const std::vector<mlir::Operation *> expected{
      reads[0], reads[1], guards[0], publications[0], observations[0], reads[2]};
  check(reached_entry && ancestors == expected,
        "second check retains read A/read B/check0/publish C/observe C/late read D prefix");
  check(std::find(ancestors.begin(), ancestors.end(), publications[1]) == ancestors.end(),
        "second validation failure frontier does not include final publication E");
  check(guards[1]->getOperand(1) == reads[2]->getResult(0) &&
            guards[1]->getOperand(2) == gemms[0]->getResult(0) &&
            gemms[1]->getOperand(1) == reads[2]->getResult(0) &&
            gemms[1]->getOperand(2) == gemms[0]->getResult(0),
        "late D and immutable carried C remain ordered noncommuting RHS operands");
  check(publications[0]->getOperand(2) == gemms[0]->getResult(0) &&
            gemms[0]->getOperand(0) == guards[0]->getResult(0),
        "first publication is data-dependent on first checked mathematical result");
  const auto epoch = publications[0]->getAttrOfType<mlir::StringAttr>("output_epoch");
  check(epoch && reads[2]->getAttr("resource_epoch") == epoch &&
            reads[2]->getOperand(0) == observations[0]->getResult(0),
        "possibly overlapping D is read after C publication and observation");

  // Stronger than a broken-token edit: produce a structurally self-consistent
  // *different* program by hoisting D's read before publication C, updating all
  // touched order and epoch references. Intrinsic validity cannot authenticate
  // this altered observable trace; source pairing must still reject it.
  mlir::OwningOpRef<mlir::ModuleOp> changed = module.clone();
  reads.clear(); guards.clear(); publications.clear(); observations.clear();
  changed->walk([&](mlir::Operation *op) {
    const auto name = op->getName().getStringRef();
    if (name == "mdsl_admission.read") reads.push_back(op);
    if (name == "mdsl_admission.check_gemm") guards.push_back(op);
    if (name == "mdsl_admission.publish") publications.push_back(op);
    if (name == "mdsl_admission.observe") observations.push_back(op);
  });
  auto *late = reads[2];
  for (unsigned index = 2; index != 4; ++index) {
    auto *dimension = late->getOperand(index).getDefiningOp();
    if (dimension) dimension->moveBefore(publications[0]);
  }
  late->moveBefore(publications[0]);
  late->setOperand(0, guards[0]->getResult(0));
  late->setAttr("resource_epoch", publications[0]->getAttr("input_epoch"));
  publications[0]->setOperand(0, late->getResult(1));
  guards[1]->setOperand(0, observations[0]->getResult(0));
  std::string error;
  check(closed_region::verifyModule(*changed, error),
        "compound hoisted-read mutation is structurally self-consistent: " + error);
  check(!frontend::verifyClosedRegionMatchesEvidence(evidence, *changed, error) && !error.empty(),
        "native source pairing rejects a self-consistent but wrongly hoisted aliased read");
}

} // namespace matcore::mdslc::test

#endif
