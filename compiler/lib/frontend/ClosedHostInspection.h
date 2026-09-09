#pragma once

#include "ClosedRegionAdmissionInternal.h"

namespace matcore::mdslc::frontend::detail {

// Noninstalled inspection callback over the same Clang preprocessing and main
// source identity checks as region admission. This never issues semantic
// evidence. The callback decides whether an actual declaration requires owned
// interface headers; ordinary header-free host code needs no forced include.
using ClosedHostInspector = std::function<bool(
    clang::ASTContext &, const clang::Decl *, clang::FileManager &,
    ClosedRegionASTPolicy &, std::string &)>;

bool inspectClosedHost(
    const std::vector<std::string> &arguments,
    const std::string &working_directory,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> filesystem,
    const std::string &source, const std::string &input,
    const ClosedHostInspector &, std::string &preprocessing_identity,
    std::string &error);
} // namespace matcore::mdslc::frontend::detail
