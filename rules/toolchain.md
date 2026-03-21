# Toolchain Rule

- Use Clang/LLVM `18.1.3` only.
- Set `CMAKE_C_COMPILER` to `/usr/bin/clang`.
- Set `CMAKE_CXX_COMPILER` to `/usr/bin/clang++`.
- Build as C++20 only.
- Use `Ninja` as the generator.
- Enable `ccache` for compiler launchers.
- Do not introduce alternate host compilers, standards, or build generators.

