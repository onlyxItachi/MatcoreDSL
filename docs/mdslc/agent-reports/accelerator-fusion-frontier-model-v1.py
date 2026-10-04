#!/usr/bin/env python3
"""MODEL ONLY: distinguish required guards from realization-dependent failures.

This imports no Matcore implementation and executes no GPU or generated kernel.
It is a small, explicit state model, not a proof of a production adapter. An
operation's ordered internal guards are abstracted to its first guard error.

Two laws are deliberately distinct:
  * sequential(): retain two independently fallible candidate invocations;
  * combined(): proposed single-realization law, with no producer invocation,
    original guard bundles at f1/f2, and the shared invocation owned by f2.

The difference in their injected-failure traces does NOT disprove the combined
law: resource-contract item 8 permits implementation failures to vary. Tests
also reject actual violations of the proposed law (skipped first guards,
second-guard hoisting, and freeing possibly live resources). No source authority
or contract amendment is issued by this model.
"""

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
import json


class DeviceOutcome(Enum):
    OK = "ok"
    KNOWN_FAILURE = "known_failure"
    UNKNOWN_COMPLETION = "unknown_completion"


OK = "ok"
F1 = 2
F2 = 3
GUARDS = (
    OK,
    "invalid_value",
    "shape_mismatch",
    "extent_overflow",
    "candidate_unavailable",
    "candidate_incompatible",
    "unsupported_fp_environment",
)


@dataclass
class State:
    # A successful earlier host publication is outside the selected pure pair.
    code: str = OK
    failed_frontier: int = 0
    completed_frontier: int = 1
    completed_effect_frontier: int = 1
    published_bytes: tuple = (0x3F800000, 0x80000000)
    output_issued: bool = False
    adapter_poisoned: bool = False
    retained_live_resources: bool = False
    report_actual: str = "none"
    report_invocation_attempted: bool = False
    trace: list = field(default_factory=lambda: ["publish@1"])

    def fail(self, code, frontier):
        if self.code == OK:
            self.code = code
            self.failed_frontier = frontier
            self.trace.append(f"fail:{code}@{frontier}")

    def guard(self, code, frontier):
        if self.code != OK:
            return False
        self.trace.append(f"guards@{frontier}")
        if code != OK:
            self.fail(code, frontier)
            return False
        return True

    def retire(self, frontier):
        if self.code == OK:
            self.completed_frontier = frontier
            self.trace.append(f"retire@{frontier}")

    def invoke(self, outcome, frontier, name):
        if self.code != OK:
            return False
        self.report_actual = name
        self.report_invocation_attempted = True
        self.trace.append(f"invoke:{name}@{frontier}")
        if outcome is DeviceOutcome.UNKNOWN_COMPLETION:
            self.adapter_poisoned = True
            self.retained_live_resources = True
            self.trace.append("quarantine+poison")
        if outcome is not DeviceOutcome.OK:
            self.fail("candidate_failure", frontier)
            return False
        self.trace.append(f"checked-completion:{name}@{frontier}")
        return True

    def later_effect(self):
        if self.code == OK:
            self.trace.append("later-effect")
            self.published_bytes = (0x40000000,)
            self.completed_effect_frontier = 4

    def observable(self):
        return (
            self.code, self.failed_frontier, self.completed_frontier,
            self.completed_effect_frontier, self.published_bytes,
            self.output_issued, self.adapter_poisoned,
            self.retained_live_resources, self.report_actual,
            self.report_invocation_attempted,
        )


def sequential(first_guard, second_guard, first_device, second_device):
    state = State()
    if not state.guard(first_guard, F1):
        return state
    if not state.invoke(first_device, F1, "producer"):
        return state
    state.retire(F1)
    if not state.guard(second_guard, F2):
        return state
    if not state.invoke(second_device, F2, "consumer"):
        return state
    state.output_issued = True
    state.retire(F2)
    return state


def combined(first_guard, second_guard, shared_device):
    state = State()
    if not state.guard(first_guard, F1):
        return state
    # Logical retirement is not a claim of materialized C or device invocation.
    state.retire(F1)
    if not state.guard(second_guard, F2):
        return state
    if not state.invoke(shared_device, F2, "combined"):
        return state
    state.output_issued = True
    state.retire(F2)
    return state


def satisfies_proposed_combined_law(state, first_guard, second_guard, outcome):
    """Finite-model predicate, NOT a verifier/issuer for a Matcore plan."""
    if (state.completed_effect_frontier != 1 or
            state.published_bytes != State().published_bytes):
        return False
    if first_guard != OK:
        return (
            state.code == first_guard and state.failed_frontier == F1 and
            state.completed_frontier == 1 and not state.output_issued and
            not state.report_invocation_attempted and
            not state.adapter_poisoned and not state.retained_live_resources and
            state.trace == ["publish@1", f"guards@{F1}",
                            f"fail:{first_guard}@{F1}"]
        )
    if second_guard != OK:
        return (
            state.code == second_guard and state.failed_frontier == F2 and
            state.completed_frontier == F1 and not state.output_issued and
            not state.report_invocation_attempted and
            not state.adapter_poisoned and not state.retained_live_resources and
            state.trace == ["publish@1", f"guards@{F1}", f"retire@{F1}",
                            f"guards@{F2}", f"fail:{second_guard}@{F2}"]
        )
    if state.trace[:4] != ["publish@1", f"guards@{F1}",
                           f"retire@{F1}", f"guards@{F2}"]:
        return False
    if (not state.report_invocation_attempted or
            state.report_actual != "combined"):
        return False
    if outcome is DeviceOutcome.OK:
        return (state.code == OK and state.failed_frontier == 0 and
                state.completed_frontier == F2 and state.output_issued and
                not state.adapter_poisoned and not state.retained_live_resources)
    unknown = outcome is DeviceOutcome.UNKNOWN_COMPLETION
    return (state.code == "candidate_failure" and
            state.failed_frontier == F2 and state.completed_frontier == F1 and
            not state.output_issued and state.adapter_poisoned == unknown and
            state.retained_live_resources == unknown)


def main():
    checks = 0

    def check(condition, name):
        nonlocal checks
        checks += 1
        if not condition:
            raise RuntimeError(f"MODEL check failed: {name}")

    scenarios = 0
    for first, second, outcome in product(GUARDS, GUARDS, DeviceOutcome):
        scenarios += 1
        state = combined(first, second, outcome)
        check(satisfies_proposed_combined_law(state, first, second, outcome),
              f"combined law: {first}/{second}/{outcome.value}")
        if state.code != OK:
            before = (state.observable(), tuple(state.trace))
            state.fail("extent_overflow", 99)
            state.later_effect()
            check((state.observable(), tuple(state.trace)) == before,
                  "actual failure is sticky and blocks later effects")

    # Conditional mismatch: this is NOT a semantic impossibility proof, because
    # the proposed combined realization removes the first invocation entirely.
    old = sequential(OK, "shape_mismatch", DeviceOutcome.KNOWN_FAILURE,
                     DeviceOutcome.OK)
    new = combined(OK, "shape_mismatch", DeviceOutcome.KNOWN_FAILURE)
    check(old.failed_frontier == F1 and old.code == "candidate_failure",
          "retained producer failure precedes consumer guard")
    check(new.failed_frontier == F2 and new.code == "shape_mismatch",
          "removed launch has no counterfactual failure obligation")
    check(old.observable() != new.observable(), "conditional trace mismatch")
    check(satisfies_proposed_combined_law(
        new, OK, "shape_mismatch", DeviceOutcome.KNOWN_FAILURE),
        "sequential mismatch does not refute realization-dependent law")

    old_unknown = sequential(OK, "shape_mismatch",
                             DeviceOutcome.UNKNOWN_COMPLETION, DeviceOutcome.OK)
    check(old_unknown.adapter_poisoned and old_unknown.retained_live_resources,
          "actual unknown completion requires quarantine")
    check(not new.adapter_poisoned and not new.retained_live_resources,
          "no actual invocation creates no live resources to quarantine")

    # Real guard-order violation, even if every device invocation is infallible.
    hoisted = State()
    hoisted.guard("shape_mismatch", F2)
    check(not satisfies_proposed_combined_law(
        hoisted, "candidate_unavailable", "shape_mismatch", DeviceOutcome.OK),
        "reject consumer guard before actual first capability failure")

    # Empty final output is not permission to drop the FULL logical C guards.
    skipped = State()
    skipped.output_issued = True
    skipped.retire(F2)
    check(not satisfies_proposed_combined_law(
        skipped, "extent_overflow", OK, DeviceOutcome.OK),
        "reject skipped first extent due to final empty result")

    # Shared failure can be owned by f2 under the proposed law, but its live
    # resources cannot be freed and no Value may escape on uncertain completion.
    uncertain = combined(OK, OK, DeviceOutcome.UNKNOWN_COMPLETION)
    check(satisfies_proposed_combined_law(
        uncertain, OK, OK, DeviceOutcome.UNKNOWN_COMPLETION),
        "shared unknown-completion failure can retain guard-only f1")
    uncertain.retained_live_resources = False
    check(not satisfies_proposed_combined_law(
        uncertain, OK, OK, DeviceOutcome.UNKNOWN_COMPLETION),
        "reject freeing resources after actual unknown completion")
    uncertain.retained_live_resources = True
    uncertain.output_issued = True
    check(not satisfies_proposed_combined_law(
        uncertain, OK, OK, DeviceOutcome.UNKNOWN_COMPLETION),
        "reject output issued after actual unknown completion")

    # An aggregate device error alone cannot reconstruct the *sequential* first
    # failed operation. This is only required if that stronger law is selected.
    failed_first = sequential(OK, OK, DeviceOutcome.KNOWN_FAILURE,
                              DeviceOutcome.OK)
    failed_second = sequential(OK, OK, DeviceOutcome.OK,
                               DeviceOutcome.KNOWN_FAILURE)
    check(failed_first.code == failed_second.code == "candidate_failure" and
          failed_first.failed_frontier != failed_second.failed_frontier,
          "one aggregate error does not prove sequential attribution")

    print(json.dumps({"evidence": "MODEL_ONLY",
                      "conditional_sequential_trace": old.trace,
                      "proposed_combined_trace": new.trace}, sort_keys=True))
    print(f"MODEL_ONLY: {scenarios} proposed-law scenarios; {checks} checks; "
          "0 failures; no GPU/runtime/source execution")


if __name__ == "__main__":
    main()
