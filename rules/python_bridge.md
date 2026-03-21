# Python Bridge Rule

- Use `nanobind` for all Python bindings.
- Accept tensor inputs as zero-copy `nb::ndarray`.
- Use `nb::gil_scoped_release` before MLIR compilation work begins.
- Do not copy tensor memory when passing data into the native boundary.
- Preserve raw pointer access and shape/stride metadata across the bridge.
- Do not replace the bridge with pybind11, ctypes, or string-based marshaling.

