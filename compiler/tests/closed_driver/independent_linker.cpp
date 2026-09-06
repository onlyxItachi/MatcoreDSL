// Deliberately not a linker. A driver must not execute this PATH-selected tool.
#include <fstream>
#include <string>

int main(int argc, char **argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "-o") {
      std::ofstream out(argv[i + 1], std::ios::binary);
      out << "independent fake linker accepted by inherited PATH\n";
      return out ? 0 : 2;
    }
  }
  return 3;
}
