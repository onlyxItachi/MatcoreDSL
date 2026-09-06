#include "ClosedRegionHostInputs.h"

#include "../support/platform_support.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace matcore::mdslc::frontend::closed_region_host {
namespace {
namespace fs = std::filesystem;
namespace vfs = llvm::vfs;
constexpr std::size_t maximum_queries = 32768;
constexpr std::size_t maximum_file_bytes = 16U * 1024U * 1024U;
constexpr std::size_t maximum_total_bytes = 64U * 1024U * 1024U;

std::error_code denied() {
  return std::make_error_code(std::errc::operation_not_permitted);
}

std::string digest(llvm::StringRef bytes) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes)), true);
}

void appendBoundField(std::string &identity, llvm::StringRef value) {
  identity += std::to_string(value.size()) + ":" + value.str();
}

bool sameSnapshot(const support::FileSnapshotV1 &a,
                  const support::FileSnapshotV1 &b) {
  return a.version == b.version && a.normalized_path == b.normalized_path &&
         a.exists == b.exists && a.regular_file == b.regular_file &&
         a.size_bytes == b.size_bytes &&
         a.last_write_time_ticks == b.last_write_time_ticks &&
         a.identity == b.identity && a.path_identity_chain == b.path_identity_chain;
}

bool sameStatus(const vfs::Status &a, const vfs::Status &b) {
  return a.getName() == b.getName() && a.getUniqueID() == b.getUniqueID() &&
         a.getType() == b.getType() && a.getPermissions() == b.getPermissions() &&
         a.getLastModificationTime() == b.getLastModificationTime() &&
         a.getUser() == b.getUser() && a.getGroup() == b.getGroup() &&
         a.getSize() == b.getSize();
}

std::string absoluteLookup(llvm::StringRef path, llvm::StringRef cwd) {
  // Do not lexical-normalize: link/../header traverses the link before '..'.
  const fs::path value(path.str());
  return value.is_absolute() ? value.string() : (fs::path(cwd.str()) / value).string();
}

bool environmentOkay(std::string &error) {
  if (auto name = support::poisoned_compiler_environment_v1(error)) {
    error = "inherited compiler control variable is forbidden: " + *name;
    return false;
  }
  if (!error.empty()) return false;
  constexpr std::array extra{
      "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
      "SDKROOT", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH",
      "SOURCE_DATE_EPOCH", "MACOSX_DEPLOYMENT_TARGET", "IPHONEOS_DEPLOYMENT_TARGET"};
  for (const char *name : extra) {
    const auto value = support::environment_utf8_v1(name, error);
    if (!error.empty()) return false;
    if (value) {
      error = std::string("inherited host-context input is forbidden: ") + name;
      return false;
    }
  }
  return true;
}

struct BinaryInput {
  std::string path;
  support::FileSnapshotV1 snapshot;
  std::string sha256;
};

std::optional<BinaryInput> binaryInput(const std::string &path, std::string &error) {
  const auto before = support::capture_file_snapshot_v1(path, error);
  if (!error.empty() || !before.regular_file || !before.identity ||
      before.size_bytes > 512U * 1024U * 1024U) {
    if (error.empty()) error = "configured compiler input is not a bounded regular file: " + path;
    return {};
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) { error = "cannot read configured compiler input: " + path; return {}; }
  llvm::SHA256 hash;
  std::array<char, 65536> chunk{};
  std::uint64_t count = 0;
  while (stream) {
    stream.read(chunk.data(), chunk.size());
    const auto size = stream.gcount();
    if (size > 0) {
      count += static_cast<std::uint64_t>(size);
      hash.update(llvm::StringRef(chunk.data(), static_cast<std::size_t>(size)));
    }
  }
  if (!stream.eof() || count != before.size_bytes) {
    error = "configured compiler input changed or failed while reading: " + path;
    return {};
  }
  const auto after = support::capture_file_snapshot_v1(path, error);
  if (!error.empty() || !sameSnapshot(before, after)) {
    if (error.empty()) error = "configured compiler input changed while reading: " + path;
    return {};
  }
  return BinaryInput{path, after, llvm::toHex(hash.final(), true)};
}

struct StatusRecord {
  std::optional<vfs::Status> value;
  std::error_code error;
  support::FileSnapshotV1 snapshot;
  bool owned = false;
  // The shared snapshot helper stops at an absent leaf. Retain all ancestor
  // outcomes too, so a missing search path cannot be silently retargeted.
  std::vector<support::FileSnapshotV1> absent_ancestors;
};
struct OpenRecord {
  std::optional<vfs::Status> status;
  std::string bytes;
  std::error_code error;
};
struct RealPathRecord { std::string path; std::error_code error; };
struct DirectoryRecord {
  std::vector<vfs::directory_entry> entries;
  std::error_code error;
};
struct LocalRecord { bool local = false; std::error_code error; };
struct Records {
  std::string cwd, input, source;
  std::vector<std::string> arguments;
  std::vector<BinaryInput> binaries;
  std::map<std::string, StatusRecord> status;
  std::map<std::string, OpenRecord> opens;
  std::map<std::string, RealPathRecord> realpaths;
  std::map<std::string, DirectoryRecord> directories;
  std::map<std::string, LocalRecord> locality;
  std::set<std::string> owned;
  std::string error;
  std::size_t bytes = 0;
};

std::vector<support::FileSnapshotV1>
absentAncestors(const std::string &path, std::string &error) {
  std::vector<support::FileSnapshotV1> result;
  fs::path parent = fs::path(path).parent_path();
  while (!parent.empty()) {
    if (result.size() >= 128) {
      error = "negative lookup exceeds the bounded path-ancestry depth"; return {};
    }
    result.push_back(support::capture_file_snapshot_v1(parent, error));
    if (!error.empty()) return {};
    const auto next = parent.parent_path();
    if (next == parent) break;
    parent = next;
  }
  return result;
}

bool sameAncestors(const std::vector<support::FileSnapshotV1> &a,
                   const std::vector<support::FileSnapshotV1> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t index = 0; index < a.size(); ++index) {
    // Parent-directory timestamps change for unrelated sibling files. Only
    // namespace traversal identity matters for this extra negative-path guard.
    const auto &left = a[index];
    const auto &right = b[index];
    if (left.normalized_path != right.normalized_path || left.exists != right.exists ||
        left.regular_file != right.regular_file || left.identity != right.identity ||
        left.path_identity_chain != right.path_identity_chain) return false;
  }
  return true;
}

class FrozenFile final : public vfs::File {
public:
  explicit FrozenFile(const OpenRecord &record) : record_(record) {}
  llvm::ErrorOr<vfs::Status> status() override { return *record_.status; }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const llvm::Twine &name, int64_t, bool, bool) override {
    return llvm::MemoryBuffer::getMemBufferCopy(record_.bytes, name.str());
  }
  std::error_code close() override { return {}; }
private:
  OpenRecord record_;
};

class FrozenDirectory final : public vfs::detail::DirIterImpl {
public:
  explicit FrozenDirectory(std::vector<vfs::directory_entry> entries)
      : entries_(std::move(entries)) {
    if (!entries_.empty()) CurrentEntry = entries_.front();
  }
  std::error_code increment() override {
    ++next_;
    CurrentEntry = next_ < entries_.size() ? entries_[next_] : vfs::directory_entry();
    return {};
  }
private:
  std::vector<vfs::directory_entry> entries_;
  std::size_t next_ = 0;
};

DirectoryRecord readDirectory(vfs::FileSystem &filesystem, const std::string &path) {
  DirectoryRecord result;
  auto entry = filesystem.dir_begin(path, result.error);
  const vfs::directory_iterator end;
  while (!result.error && entry != end) {
    if (result.entries.size() >= maximum_queries) {
      result.error = std::make_error_code(std::errc::value_too_large);
      break;
    }
    result.entries.push_back(*entry);
    entry.increment(result.error);
  }
  return result;
}

bool sameDirectory(const DirectoryRecord &a, const DirectoryRecord &b) {
  if (a.error != b.error || a.entries.size() != b.entries.size()) return false;
  for (std::size_t index = 0; index < a.entries.size(); ++index)
    if (a.entries[index].path() != b.entries[index].path() ||
        a.entries[index].type() != b.entries[index].type()) return false;
  return true;
}

bool recordsUnchanged(const Records &records, std::string &error) {
  error.clear();
  if (!environmentOkay(error)) return false;
  auto physical = vfs::createPhysicalFileSystem();
  if (const auto ec = physical->setCurrentWorkingDirectory(records.cwd)) {
    error = "cannot recheck captured working directory: " + ec.message(); return false;
  }
  for (const auto &[path, record] : records.status) {
    if (record.owned) continue;
    const auto now = support::capture_file_snapshot_v1(path, error);
    if (!error.empty() || !sameSnapshot(record.snapshot, now)) {
      error = "captured path identity or metadata changed: " + path; return false;
    }
    if (!record.snapshot.exists) {
      const auto ancestors = absentAncestors(path, error);
      if (!error.empty() || !sameAncestors(record.absent_ancestors, ancestors)) {
        error = "captured negative-lookup path ancestry changed: " + path; return false;
      }
    }
    const auto status = physical->status(path);
    if (record.value ? (!status || !sameStatus(*record.value, *status))
                     : (status || status.getError() != record.error)) {
      error = "captured filesystem status outcome changed: " + path; return false;
    }
  }
  for (const auto &[path, record] : records.opens) {
    if (records.owned.count(path)) continue;
    auto opened = physical->openFileForRead(path);
    if (!record.status) {
      if (opened || opened.getError() != record.error) {
        error = "captured file-open outcome changed: " + path; return false;
      }
      continue;
    }
    if (!opened) { error = "captured file cannot be reopened: " + path; return false; }
    const auto status = (*opened)->status();
    auto buffer = (*opened)->getBuffer(path);
    const auto close = (*opened)->close();
    if (!status || !sameStatus(*record.status, *status) || !buffer || close ||
        (*buffer)->getBuffer() != record.bytes) {
      error = "captured source bytes or open identity changed: " + path; return false;
    }
    const auto after = support::capture_file_snapshot_v1(path, error);
    const auto before = records.status.find(path);
    if (!error.empty() || before == records.status.end() ||
        !sameSnapshot(before->second.snapshot, after)) {
      error = "captured source changed during final recheck: " + path; return false;
    }
  }
  for (const auto &[path, record] : records.realpaths) {
    if (records.owned.count(path)) continue;
    llvm::SmallString<256> output;
    const auto ec = physical->getRealPath(path, output);
    if (ec != record.error || (!ec && output != record.path)) {
      error = "captured realpath lookup changed: " + path; return false;
    }
  }
  for (const auto &[path, record] : records.directories) {
    if (records.owned.count(path)) continue;
    if (!sameDirectory(record, readDirectory(*physical, path))) {
      error = "captured directory lookup changed: " + path; return false;
    }
  }
  for (const auto &[path, record] : records.locality) {
    if (records.owned.count(path)) continue;
    bool local = false;
    const auto ec = physical->isLocal(path, local);
    if (ec != record.error || (!ec && local != record.local)) {
      error = "captured locality lookup changed: " + path; return false;
    }
  }
  for (const auto &binary : records.binaries) {
    const auto now = binaryInput(binary.path, error);
    if (!now || !sameSnapshot(binary.snapshot, now->snapshot) ||
        binary.sha256 != now->sha256) {
      error = "configured compiler input changed: " + binary.path; return false;
    }
  }
  return true;
}

std::string identityOf(const Records &records) {
  std::string text;
  appendBoundField(text, "mdsl-private-host-context-v1");
  appendBoundField(text, records.cwd); appendBoundField(text, records.input); appendBoundField(text, digest(records.source));
  appendBoundField(text, nativeClangRuntimeVersionV1());
  appendBoundField(text, "arguments"); appendBoundField(text, std::to_string(records.arguments.size()));
  for (const auto &argument : records.arguments) appendBoundField(text, argument);
  appendBoundField(text, "binaries"); appendBoundField(text, std::to_string(records.binaries.size()));
  for (const auto &binary : records.binaries) {
    appendBoundField(text, binary.path); appendBoundField(text, binary.sha256);
    appendBoundField(text, std::to_string(binary.snapshot.identity.words[0]));
    appendBoundField(text, std::to_string(binary.snapshot.identity.words[1]));
    appendBoundField(text, std::to_string(binary.snapshot.last_write_time_ticks));
    appendBoundField(text, std::to_string(binary.snapshot.path_identity_chain.size()));
    for (const auto &component : binary.snapshot.path_identity_chain) {
      appendBoundField(text, std::to_string(component.identity.words[0]));
      appendBoundField(text, std::to_string(component.identity.words[1]));
      for (const auto word : component.metadata_words) appendBoundField(text, std::to_string(word));
      appendBoundField(text, component.symbolic_link_target);
    }
  }
  appendBoundField(text, "statuses"); appendBoundField(text, std::to_string(records.status.size()));
  for (const auto &[path, record] : records.status) {
    appendBoundField(text, "status"); appendBoundField(text, path);
    appendBoundField(text, record.owned ? "owned" : "physical");
    appendBoundField(text, record.value ? "present" : "absent");
    appendBoundField(text, record.error.category().name()); appendBoundField(text, std::to_string(record.error.value()));
    if (record.value) {
      const auto &s = *record.value;
      appendBoundField(text, s.getName());
      appendBoundField(text, std::to_string(s.getUniqueID().getDevice()));
      appendBoundField(text, std::to_string(s.getUniqueID().getFile()));
      appendBoundField(text, std::to_string(s.getSize()));
      appendBoundField(text, std::to_string(s.getUser()));
      appendBoundField(text, std::to_string(s.getGroup()));
      appendBoundField(text, std::to_string(s.getLastModificationTime().time_since_epoch().count()));
      appendBoundField(text, std::to_string(static_cast<unsigned>(s.getType())));
      appendBoundField(text, std::to_string(static_cast<unsigned>(s.getPermissions())));
    }
    appendBoundField(text, std::to_string(record.snapshot.path_identity_chain.size()));
    for (const auto &component : record.snapshot.path_identity_chain) {
      appendBoundField(text, std::to_string(component.identity.words[0]));
      appendBoundField(text, std::to_string(component.identity.words[1]));
      for (const auto word : component.metadata_words) appendBoundField(text, std::to_string(word));
      appendBoundField(text, component.symbolic_link_target);
    }
    appendBoundField(text, std::to_string(record.absent_ancestors.size()));
    for (const auto &ancestor : record.absent_ancestors) {
      appendBoundField(text, ancestor.normalized_path.string());
      appendBoundField(text, ancestor.exists ? "present" : "absent");
      appendBoundField(text, std::to_string(ancestor.identity.words[0]));
      appendBoundField(text, std::to_string(ancestor.identity.words[1]));
      appendBoundField(text, std::to_string(ancestor.path_identity_chain.size()));
      for (const auto &component : ancestor.path_identity_chain) {
        appendBoundField(text, std::to_string(component.identity.words[0]));
        appendBoundField(text, std::to_string(component.identity.words[1]));
        for (const auto word : component.metadata_words) appendBoundField(text, std::to_string(word));
        appendBoundField(text, component.symbolic_link_target);
      }
    }
  }
  appendBoundField(text, "opens"); appendBoundField(text, std::to_string(records.opens.size()));
  for (const auto &[path, record] : records.opens) {
    appendBoundField(text, "open"); appendBoundField(text, path); appendBoundField(text, digest(record.bytes));
    appendBoundField(text, record.error.category().name()); appendBoundField(text, std::to_string(record.error.value()));
  }
  appendBoundField(text, "realpaths"); appendBoundField(text, std::to_string(records.realpaths.size()));
  for (const auto &[path, record] : records.realpaths) {
    appendBoundField(text, "realpath"); appendBoundField(text, path); appendBoundField(text, record.path);
    appendBoundField(text, record.error.category().name()); appendBoundField(text, std::to_string(record.error.value()));
  }
  appendBoundField(text, "directories"); appendBoundField(text, std::to_string(records.directories.size()));
  for (const auto &[path, record] : records.directories) {
    appendBoundField(text, "directory"); appendBoundField(text, path);
    appendBoundField(text, record.error.category().name()); appendBoundField(text, std::to_string(record.error.value()));
    appendBoundField(text, std::to_string(record.entries.size()));
    for (const auto &entry : record.entries) {
      appendBoundField(text, entry.path()); appendBoundField(text, std::to_string(static_cast<unsigned>(entry.type())));
    }
  }
  appendBoundField(text, "locality"); appendBoundField(text, std::to_string(records.locality.size()));
  for (const auto &[path, record] : records.locality) {
    appendBoundField(text, "local"); appendBoundField(text, path); appendBoundField(text, record.local ? "yes" : "no");
    appendBoundField(text, record.error.category().name()); appendBoundField(text, std::to_string(record.error.value()));
  }
  return "sha256:" + digest(text);
}
} // namespace

struct HostInputReplay::Audit { std::string error; };

namespace {
class InputFileSystem final : public vfs::FileSystem {
public:
  explicit InputFileSystem(std::shared_ptr<Records> records)
      : records_(std::move(records)), physical_(vfs::createPhysicalFileSystem()) {
    if (auto ec = physical_->setCurrentWorkingDirectory(records_->cwd))
      fail("cannot select capture working directory: " + ec.message());
  }
  InputFileSystem(std::shared_ptr<Records> records, std::shared_ptr<HostInputReplay::Audit> audit)
      : records_(std::move(records)), audit_(std::move(audit)) {}

  llvm::ErrorOr<vfs::Status> status(const llvm::Twine &lookup) override {
    const auto path = key(lookup);
    auto found = records_->status.find(path);
    if (found == records_->status.end()) {
      if (!physical_) { miss("status", path); return denied(); }
      if (!budget()) return denied();
      std::string error;
      const auto before = support::capture_file_snapshot_v1(path, error);
      if (!error.empty()) { fail(error + ": " + path); return denied(); }
      const auto ancestors = before.exists ? std::vector<support::FileSnapshotV1>{}
                                           : absentAncestors(path, error);
      if (!error.empty()) { fail(error + ": " + path); return denied(); }
      const auto observed = physical_->status(path);
      const auto after = support::capture_file_snapshot_v1(path, error);
      const auto later_ancestors = after.exists ? std::vector<support::FileSnapshotV1>{}
                                               : absentAncestors(path, error);
      const bool identity_matches = !observed ||
          after.identity.kind != support::FileIdentityKindV1::posix_device_inode ||
          (after.identity.words[0] == observed->getUniqueID().getDevice() &&
           after.identity.words[1] == observed->getUniqueID().getFile());
      if (!error.empty() || !sameSnapshot(before, after) ||
          !sameAncestors(ancestors, later_ancestors) ||
          static_cast<bool>(observed) != after.exists || !identity_matches) {
        fail("path changed during status capture: " + path); return denied();
      }
      StatusRecord record;
      record.snapshot = after;
      record.absent_ancestors = ancestors;
      if (observed) record.value = *observed;
      else record.error = observed.getError();
      found = records_->status.emplace(path, std::move(record)).first;
    }
    if (found->second.value) return *found->second.value;
    return found->second.error;
  }

  bool exists(const llvm::Twine &path) override { return static_cast<bool>(status(path)); }

  llvm::ErrorOr<std::unique_ptr<vfs::File>> openFileForRead(const llvm::Twine &lookup) override {
    const auto path = key(lookup);
    auto found = records_->opens.find(path);
    if (found == records_->opens.end()) {
      if (!physical_) { miss("open", path); return denied(); }
      if (!budget()) return denied();
      const auto expected = status(path);
      if (!records_->error.empty()) return denied();
      OpenRecord record;
      auto opened = physical_->openFileForRead(path);
      if (!opened) {
        record.error = opened.getError();
      } else {
        const auto actual = (*opened)->status();
        if (!expected || !actual || !actual->isRegularFile() ||
            !sameStatus(*expected, *actual) || actual->getSize() > maximum_file_bytes ||
            actual->getSize() > maximum_total_bytes - records_->bytes) {
          (*opened)->close();
          fail("file-open identity/type/size is outside captured context: " + path);
          return denied();
        }
        auto buffer = (*opened)->getBuffer(path);
        const auto close = (*opened)->close();
        if (!buffer || close || (*buffer)->getBufferSize() != actual->getSize()) {
          fail("file failed or changed during source read: " + path); return denied();
        }
        std::string error;
        const auto after = support::capture_file_snapshot_v1(path, error);
        if (!error.empty() || !sameSnapshot(records_->status.at(path).snapshot, after)) {
          fail("file changed during source read: " + path); return denied();
        }
        record.status = *actual;
        record.bytes = (*buffer)->getBuffer().str();
        records_->bytes += record.bytes.size();
      }
      found = records_->opens.emplace(path, std::move(record)).first;
    }
    if (!found->second.status) return found->second.error;
    return std::unique_ptr<vfs::File>(new FrozenFile(found->second));
  }

  vfs::directory_iterator dir_begin(const llvm::Twine &lookup, std::error_code &ec) override {
    const auto path = key(lookup);
    auto found = records_->directories.find(path);
    if (found == records_->directories.end()) {
      if (!physical_) { miss("directory", path); ec = denied(); return {}; }
      if (!budget()) { ec = denied(); return {}; }
      (void)status(path);
      DirectoryRecord record;
      if (records_->owned.count(path)) {
        if (!records_->status.at(path).value->isDirectory())
          record.error = std::make_error_code(std::errc::not_a_directory);
        else
          for (const auto &[child, child_status] : records_->status)
            if (child_status.owned && child_status.value &&
                fs::path(child).parent_path().string() == path)
              record.entries.emplace_back(child, child_status.value->getType());
      } else record = readDirectory(*physical_, path);
      found = records_->directories.emplace(path, std::move(record)).first;
    }
    ec = found->second.error;
    if (ec || found->second.entries.empty()) return {};
    return vfs::directory_iterator(std::make_shared<FrozenDirectory>(found->second.entries));
  }

  std::error_code setCurrentWorkingDirectory(const llvm::Twine &path) override {
    if (key(path) != records_->cwd) {
      fail("changing the authenticated working directory is forbidden: " + path.str());
      return denied();
    }
    return {};
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override { return records_->cwd; }

  std::error_code getRealPath(const llvm::Twine &lookup, llvm::SmallVectorImpl<char> &output) override {
    const auto path = key(lookup);
    auto found = records_->realpaths.find(path);
    if (found == records_->realpaths.end()) {
      if (!physical_) { miss("realpath", path); return denied(); }
      if (!budget()) return denied();
      (void)status(path);
      RealPathRecord record;
      if (records_->owned.count(path)) record.path = path;
      else {
        llvm::SmallString<256> resolved;
        record.error = physical_->getRealPath(path, resolved);
        if (!record.error) record.path = resolved.str().str();
      }
      found = records_->realpaths.emplace(path, std::move(record)).first;
    }
    if (!found->second.error) output.assign(found->second.path.begin(), found->second.path.end());
    return found->second.error;
  }

  std::error_code isLocal(const llvm::Twine &lookup, bool &result) override {
    const auto path = key(lookup);
    auto found = records_->locality.find(path);
    if (found == records_->locality.end()) {
      if (!physical_) { miss("locality", path); return denied(); }
      if (!budget()) return denied();
      (void)status(path);
      LocalRecord record;
      if (records_->owned.count(path)) record.local = true;
      else record.error = physical_->isLocal(path, record.local);
      found = records_->locality.emplace(path, record).first;
    }
    result = found->second.local;
    return found->second.error;
  }

private:
  std::string key(const llvm::Twine &path) const { return absoluteLookup(path.str(), records_->cwd); }
  void fail(const std::string &message) {
    if (audit_) { if (audit_->error.empty()) audit_->error = message; }
    else if (records_->error.empty()) records_->error = message;
  }
  void miss(const char *operation, const std::string &path) {
    fail(std::string("uncaptured replay ") + operation + " lookup: " + path);
  }
  bool budget() {
    if (records_->status.size() + records_->opens.size() + records_->realpaths.size() +
        records_->directories.size() + records_->locality.size() >= maximum_queries) {
      fail("host-context filesystem query budget exceeded"); return false;
    }
    return records_->error.empty();
  }
  std::shared_ptr<Records> records_;
  std::unique_ptr<vfs::FileSystem> physical_;
  std::shared_ptr<HostInputReplay::Audit> audit_;
};

bool configure(const Options &options, const std::string &cwd, Records &records,
               std::string &error) {
  if (!environmentOkay(error)) return false;
  const auto valid_path = [](const std::string &path) {
    if (path.empty() || path.size() > 4096 || path.find('\0') != std::string::npos) return false;
    std::string path_error;
    return support::path_from_utf8_v1(path, path_error).has_value() && path_error.empty();
  };
  std::error_code cwd_error;
  const bool directory_exists = fs::is_directory(fs::path(cwd), cwd_error);
  if (!valid_path(options.input_path) || !valid_path(cwd) ||
      !valid_path(options.clang_path) || !valid_path(options.clang_resource_directory) ||
      options.compiler_arguments.size() > 128 ||
      !fs::path(cwd).is_absolute() || cwd_error || !directory_exists) {
    error = "host inspection requires an existing absolute working directory and bounded input options";
    return false;
  }
  if (options.inspect_recovered_cpp_gemm || options.inspect_two_gemm_regions ||
      !options.trusted_public_headers.empty()) {
    error = "legacy capture selectors and public-header inputs are not part of closed host admission";
    return false;
  }
  records.cwd = cwd;
  records.input = absoluteLookup(options.input_path, cwd);
  if (!support::paths_refer_to_same_location_v1(options.clang_path,
          MDSLC_HOST_CLANG_EXECUTABLE, error) || !error.empty() ||
      options.clang_resource_directory.empty() ||
      !support::paths_refer_to_same_location_v1(options.clang_resource_directory,
          MDSLC_HOST_CLANG_RESOURCE_DIRECTORY, error) || !error.empty()) {
    error = "host inspection requires the configured exact Clang/resource tuple";
    return false;
  }
  if (!support::clang_version_matches_exact_v1(nativeClangRuntimeVersionV1(),
                                               MDSLC_HOST_TOOLCHAIN_VERSION)) {
    error = "loaded Clang runtime version is not the configured exact tuple"; return false;
  }
  const auto runtime = nativeClangRuntimeLibraryPathV1(error);
  if (!runtime || !error.empty() ||
      !support::paths_refer_to_same_location_v1(*runtime, MDSLC_HOST_CLANG_CPP_RUNTIME, error) ||
      !error.empty()) {
    error = "loaded Clang runtime identity does not match the configured tuple"; return false;
  }
  for (const std::string &path : {std::string(MDSLC_HOST_CLANG_EXECUTABLE), *runtime}) {
    auto input = binaryInput(path, error);
    if (!input) return false;
    records.binaries.push_back(std::move(*input));
  }
  records.arguments = {MDSLC_HOST_CLANG_EXECUTABLE, "-x", "c++", "-std=c++20", "-fsyntax-only", "-fno-color-diagnostics",
                       "--no-default-config", "-fno-modules", "-fno-delayed-template-parsing",
                       "-resource-dir=" MDSLC_HOST_CLANG_RESOURCE_DIRECTORY};
  for (std::size_t index = 0; index < options.compiler_arguments.size(); ++index) {
    const std::string &argument = options.compiler_arguments[index];
    if (argument.empty() || argument.size() > 4096 || argument.find('\0') != std::string::npos ||
        support::classify_untrusted_compiler_argument_v1(argument, false) !=
          support::CompilerArgumentRiskV1::none) {
      error = "unsafe host-context compiler argument: " + argument; return false;
    }
    if (argument == "-std=c++20") continue;
    const bool separate = argument == "-I" || argument == "-isystem" ||
                          argument == "-iquote" || argument == "-D" || argument == "-U";
    if (separate) {
      if (++index == options.compiler_arguments.size() ||
          options.compiler_arguments[index].empty() || options.compiler_arguments[index].size() > 4096 ||
          !support::compiler_consumed_value_is_safe_v1(options.compiler_arguments[index]) ||
          options.compiler_arguments[index].find('\0') != std::string::npos) {
        error = "invalid value for bounded host-context option: " + argument; return false;
      }
      records.arguments.push_back(argument);
      records.arguments.push_back(options.compiler_arguments[index]);
    } else if ((argument.starts_with("-I") || argument.starts_with("-D") || argument.starts_with("-U")) &&
               argument.size() > 2 && support::compiler_consumed_value_is_safe_v1(argument.substr(2))) {
      records.arguments.push_back(argument);
    } else {
      error = "compiler option is outside the bounded host-context allowlist: " + argument;
      return false;
    }
  }
  return true;
}

bool addOwnedFiles(Records &records, const OwnedHostFiles &owned, std::string &error) {
  if (owned.empty() || owned.size() > 4) {
    error = "compiler-owned virtual input count is outside the bounded contract"; return false;
  }
  std::uint64_t ordinal = 1;
  for (const auto &[path, bytes] : owned) {
    if (!path.starts_with("/__mdsl_private__/") || bytes.size() > maximum_file_bytes ||
        bytes.size() > maximum_total_bytes - records.bytes ||
        fs::path(path).lexically_normal().string() != path ||
        records.owned.count(path) || path.find('\0') != std::string::npos) {
      error = "invalid compiler-owned virtual input"; return false;
    }
    const vfs::Status status(path, llvm::sys::fs::UniqueID(std::numeric_limits<std::uint64_t>::max(), ordinal++),
                            llvm::sys::TimePoint<>(), 0, 0, bytes.size(),
                            llvm::sys::fs::file_type::regular_file, llvm::sys::fs::perms::all_read);
    records.status[path] = StatusRecord{status, {}, {}, true, {}};
    records.opens[path] = OpenRecord{status, bytes, {}};
    records.owned.insert(path);
    records.bytes += bytes.size();
  }
  const std::string directory = "/__mdsl_private__";
  const vfs::Status status(directory, llvm::sys::fs::UniqueID(std::numeric_limits<std::uint64_t>::max(), 0),
                          llvm::sys::TimePoint<>(), 0, 0, 0,
                          llvm::sys::fs::file_type::directory_file, llvm::sys::fs::perms::all_read);
  records.status[directory] = StatusRecord{status, {}, {}, true, {}};
  records.owned.insert(directory);
  return true;
}
} // namespace

struct HostInputSnapshot::Impl { Records records; std::string identity; };
struct HostInputCapture::Impl {
  std::shared_ptr<Records> records;
  llvm::IntrusiveRefCntPtr<InputFileSystem> filesystem;
};

bool HostInputReplay::ok(std::string &error) const {
  if (!audit || !filesystem) { error = "missing replay session"; return false; }
  if (!audit->error.empty()) { error = audit->error; return false; }
  return true;
}

HostInputSnapshot::HostInputSnapshot(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
const std::string &HostInputSnapshot::sourceSnapshot() const { return impl_->records.source; }
const std::string &HostInputSnapshot::inputPath() const { return impl_->records.input; }
const std::string &HostInputSnapshot::workingDirectory() const { return impl_->records.cwd; }
const std::vector<std::string> &HostInputSnapshot::arguments() const { return impl_->records.arguments; }
const std::string &HostInputSnapshot::identity() const { return impl_->identity; }
HostInputReplay HostInputSnapshot::replay() const {
  auto audit = std::make_shared<HostInputReplay::Audit>();
  (void)environmentOkay(audit->error);
  if (audit->error.empty() &&
      !support::clang_version_matches_exact_v1(nativeClangRuntimeVersionV1(),
                                               MDSLC_HOST_TOOLCHAIN_VERSION))
    audit->error = "replay Clang runtime is not the configured exact tuple";
  return {llvm::IntrusiveRefCntPtr<vfs::FileSystem>(
              new InputFileSystem(std::make_shared<Records>(impl_->records), audit)), audit};
}
bool HostInputSnapshot::unchanged(std::string &error) const { return recordsUnchanged(impl_->records, error); }

HostInputCapture::HostInputCapture(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HostInputCapture::~HostInputCapture() = default;
const std::string &HostInputCapture::sourceSnapshot() const { return impl_->records->source; }
const std::string &HostInputCapture::inputPath() const { return impl_->records->input; }
const std::string &HostInputCapture::workingDirectory() const { return impl_->records->cwd; }
const std::vector<std::string> &HostInputCapture::arguments() const { return impl_->records->arguments; }
llvm::IntrusiveRefCntPtr<vfs::FileSystem> HostInputCapture::fileSystem() const { return impl_->filesystem; }
std::shared_ptr<const HostInputSnapshot> HostInputCapture::freeze(std::string &error) {
  if (!impl_->records->error.empty()) { error = impl_->records->error; return {}; }
  if (!recordsUnchanged(*impl_->records, error)) return {};
  auto frozen = std::make_shared<HostInputSnapshot::Impl>();
  frozen->records = *impl_->records;
  frozen->identity = identityOf(frozen->records);
  return std::shared_ptr<const HostInputSnapshot>(new HostInputSnapshot(std::move(frozen)));
}

std::unique_ptr<HostInputCapture>
prepareHostInputs(const Options &options, const std::string &working_directory,
                  const OwnedHostFiles &owned_files, std::string &error) {
  error.clear();
  auto records = std::make_shared<Records>();
  if (!configure(options, working_directory, *records, error) ||
      !addOwnedFiles(*records, owned_files, error)) return {};
  if (!records->owned.count("/__mdsl_private__/fixture.h")) {
    error = "the fixed compiler-owned declaration header is required"; return {};
  }
  if (records->owned.count(records->input)) {
    error = "physical main source cannot alias a compiler-owned virtual input"; return {};
  }
  records->arguments.insert(records->arguments.end(),
      {"-include", "/__mdsl_private__/fixture.h", records->input});
  auto impl = std::make_unique<HostInputCapture::Impl>();
  impl->records = records;
  impl->filesystem = llvm::IntrusiveRefCntPtr<InputFileSystem>(new InputFileSystem(records));
  for (const std::string &path : {std::string(MDSLC_HOST_CLANG_RESOURCE_DIRECTORY),
           std::string(MDSLC_HOST_CLANG_RESOURCE_DIRECTORY) + "/include"}) {
    const auto status = impl->filesystem->status(path);
    if (!status || !status->isDirectory() || !records->error.empty()) {
      error = records->error.empty() ? "configured resource directory is unavailable"
                                     : records->error;
      return {};
    }
  }
  auto source = impl->filesystem->getBufferForFile(records->input);
  if (!source || !records->error.empty() || (*source)->getBufferSize() > 1024U * 1024U) {
    error = records->error.empty() ? "main source is unreadable or exceeds the admission byte limit"
                                   : records->error;
    return {};
  }
  records->source = (*source)->getBuffer().str();
  if (records->source.find('\0') != std::string::npos) {
    error = "main source contains a NUL byte"; return {};
  }
  return std::unique_ptr<HostInputCapture>(new HostInputCapture(std::move(impl)));
}

} // namespace matcore::mdslc::frontend::closed_region_host
