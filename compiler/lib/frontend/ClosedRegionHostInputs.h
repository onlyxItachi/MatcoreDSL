#pragma once

// Private, noninstalled compiler-input capture for the inspection-only closed
// region experiment. These records do not authenticate any semantic operation.
#include "frontend.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace matcore::mdslc::frontend::closed_region_host {

using OwnedHostFiles = std::vector<std::pair<std::string, std::string>>;

struct HostInputReplay {
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> filesystem;
  struct Audit;
  std::shared_ptr<Audit> audit;
  bool ok(std::string &error) const;
};

class HostInputSnapshot {
public:
  const std::string &sourceSnapshot() const;
  const std::string &inputPath() const;
  const std::string &workingDirectory() const;
  const std::vector<std::string> &arguments() const;
  const std::string &identity() const;
  HostInputReplay replay() const;
  bool unchanged(std::string &error) const;

private:
  struct Impl;
  explicit HostInputSnapshot(std::shared_ptr<const Impl>);
  std::shared_ptr<const Impl> impl_;
  friend class HostInputCapture;
};

class HostInputCapture {
public:
  ~HostInputCapture();
  const std::string &sourceSnapshot() const;
  const std::string &inputPath() const;
  const std::string &workingDirectory() const;
  const std::vector<std::string> &arguments() const;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fileSystem() const;
  std::shared_ptr<const HostInputSnapshot> freeze(std::string &error);

private:
  struct Impl;
  explicit HostInputCapture(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend std::unique_ptr<HostInputCapture>
  prepareHostInputs(const Options &, const std::string &, const OwnedHostFiles &,
                    std::string &);
};

std::unique_ptr<HostInputCapture>
prepareHostInputs(const Options &options, const std::string &working_directory,
                  const OwnedHostFiles &owned_files,
                  std::string &error);

} // namespace matcore::mdslc::frontend::closed_region_host
