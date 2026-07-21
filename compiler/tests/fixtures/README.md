# MDSLC validation fixtures

These files are inputs, not claims of implemented support. The authoritative
state of every case is `../integration/validation_matrix.json`:

- `active` cases are executable against the current integration milestone;
- `pending` cases specify a later acceptance requirement and are not counted
  as passes;
- `known_failure` cases capture a reproduced defect and are not accepted.

`output_contracts.mdsl` is selected with one `TEST_*` definition per run. The
rank, dtype, layout, residency, and malformed-IR cases require the runtime/IR
verifier boundary and remain pending until that boundary is integrated.
