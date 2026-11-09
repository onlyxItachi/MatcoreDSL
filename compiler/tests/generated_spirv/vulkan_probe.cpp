// Test-only process. No runtime linking, source admission, shader imports or fallback.
#include "MatcoreSpirvGemmCandidate.h"
#include "strict_oracle.h"
#include "llvm/Support/SHA256.h"
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace issuer = matcore::mdslc::spirv_candidate;
using Word = strict_oracle::Word;
struct Refusal : std::runtime_error { using std::runtime_error::runtime_error; };
static std::string quoted(std::string_view s) {
  std::string out = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
    else out += c;
  }
  return out + '"';
}
template <class T> static std::string hex(const T &bytes) {
  std::string out;
  constexpr char digits[] = "0123456789abcdef";
  for (unsigned char c : bytes) { out += digits[c >> 4]; out += digits[c & 15]; }
  return out;
}
static std::string digest(const std::vector<Word> &words) {
  return hex(llvm::SHA256::hash({reinterpret_cast<const std::uint8_t *>(words.data()), words.size() * 4}));
}
static std::string triple(const std::array<std::uint32_t, 3> &v) {
  return "[" + std::to_string(v[0]) + ',' + std::to_string(v[1]) + ',' + std::to_string(v[2]) + ']';
}
struct Report {
  std::ostringstream fields;
  std::uint64_t dispatches = 0, completions = 0, outputs = 0, inputs = 0, canaries = 0, mismatches = 0;
  void field(std::string_view key, const std::string &json) { fields << ',' << quoted(key) << ':' << json; }
  int finish(std::string_view classification, std::string_view status, std::string_view reason, int code) {
    std::ostringstream out;
    out << "{\"classification\":" << quoted(classification) << ",\"status\":" << quoted(status)
        << ",\"reason\":" << quoted(reason) << ",\"source_runtime_authority\":false"
        << ",\"process_only_qualification\":true,\"strict_contract_qualified\":false"
        << ",\"nan_policy\":\"NaN class only; all other results bit-exact\""
        << ",\"oracle_uses_host_fp_state\":false,\"dispatches\":" << dispatches
        << ",\"checked_completions\":" << completions << ",\"output_checks\":" << outputs
        << ",\"input_immutability_checks\":" << inputs << ",\"canary_checks\":" << canaries
        << ",\"mismatches\":" << mismatches << fields.str() << "}\n";
    std::fputs(out.str().c_str(), stdout); std::fflush(stdout);
    return code;
  }
  [[noreturn]] void fatal(const char *phase, VkResult result) noexcept {
    try {
      finish("ERROR", "VULKAN_API_ERROR", std::string(phase) + "; VkResult=" + std::to_string(result), 1);
    } catch (...) {
      std::fputs("Vulkan failure; receipt allocation failed; terminating without cleanup\n", stderr);
      std::fflush(stderr);
    }
    // A failed submit/wait never unwinds or frees potentially live resources.
    std::_Exit(1);
  }
  void check(VkResult result, const char *phase) { if (result != VK_SUCCESS) fatal(phase, result); }
};
// Dynamic loading occurs only after software expectations and issued shaders freeze.
#define VK_FUNCTIONS(X) \
  X(EnumerateInstanceVersion) X(CreateInstance) X(DestroyInstance) \
  X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties2) \
  X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) \
  X(CreateDevice) X(DestroyDevice) X(GetDeviceQueue) X(CreateShaderModule) \
  X(DestroyShaderModule) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
  X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
  X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
  X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(ResetCommandPool) \
  X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdPipelineBarrier) X(CmdBindPipeline) \
  X(CmdBindDescriptorSets) X(CmdDispatch) X(CreateFence) X(DestroyFence) X(ResetFences) \
  X(QueueSubmit) X(WaitForFences) X(CreateBuffer) X(DestroyBuffer) \
  X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory)
struct Vulkan {
  void *library = nullptr;
#define DECLARE(name) PFN_vk##name name = nullptr;
  VK_FUNCTIONS(DECLARE)
#undef DECLARE
  Vulkan() {
    library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) throw Refusal("Vulkan loader unavailable");
#define LOAD(name) name = reinterpret_cast<PFN_vk##name>(dlsym(library, "vk" #name)); \
    if (!name) throw Refusal("Vulkan loader missing vk" #name);
    VK_FUNCTIONS(LOAD)
#undef LOAD
  }
  ~Vulkan() { if (library) dlclose(library); }
};
struct Options {
  bool inventory = false, oracle_only = false;
  std::uint32_t device = 0;
  issuer::StaticGemmShapeV1 shape{0,0,0};
};
static std::uint64_t decimal(std::string_view text) {
  std::uint64_t result = 0;
  auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw Refusal("expected unsigned decimal argument");
  return result;
}
static Options parse(int argc, char **argv) {
  Options out;
  if (argc == 2 && std::string_view(argv[1]) == "--inventory") { out.inventory = true; return out; }
  if (argc == 2 && std::string_view(argv[1]) == "--oracle-only") { out.oracle_only = true; return out; }
  bool device = false, shape = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option(argv[i]);
    if (++i == argc) throw Refusal("missing option argument");
    const std::string_view argument(argv[i]);
    if (option == "--device" && !device) {
      const auto index = decimal(argument);
      if (index > UINT32_MAX) throw Refusal("device index overflow");
      out.device = index; device = true;
    } else if (option == "--shape" && !shape) {
      const auto first = argument.find(','), second = argument.find(',', first == argument.npos ? 0 : first + 1);
      if (first == argument.npos || second == argument.npos || argument.find(',', second + 1) != argument.npos)
        throw Refusal("shape must be M,K,N");
      std::array<std::uint64_t, 3> dims{decimal(argument.substr(0, first)), decimal(argument.substr(first + 1, second - first - 1)), decimal(argument.substr(second + 1))};
      for (auto d : dims) if (d > INT64_MAX) throw Refusal("shape exceeds signed dimension range");
      out.shape = {static_cast<std::int64_t>(dims[0]), static_cast<std::int64_t>(dims[1]), static_cast<std::int64_t>(dims[2])};
      shape = true;
    } else throw Refusal("unknown or duplicate option; use --inventory or --device N --shape M,K,N");
  }
  if (!device || !shape) throw Refusal("explicit --device and --shape required");
  return out;
}
struct DeviceInfo {
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceFloatControlsProperties fp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
  VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
};
static DeviceInfo info(Vulkan &vk, VkPhysicalDevice device) {
  DeviceInfo out;
  out.fp.pNext = &out.driver; out.driver.pNext = &out.ids;
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &out.fp; vk.GetPhysicalDeviceProperties2(device, &properties);
  out.properties = properties.properties;
  out.fp.pNext = out.driver.pNext = nullptr;
  return out;
}
static std::string deviceJson(unsigned index, const DeviceInfo &d) {
  std::ostringstream out;
  out << "{\"index\":" << index << ",\"name\":" << quoted(d.properties.deviceName)
      << ",\"vendor_id\":" << d.properties.vendorID << ",\"device_id\":" << d.properties.deviceID
      << ",\"device_type\":" << d.properties.deviceType << ",\"api_version\":" << d.properties.apiVersion
      << ",\"driver_version\":" << d.properties.driverVersion << ",\"driver_id\":" << d.driver.driverID
      << ",\"driver_name\":" << quoted(d.driver.driverName) << ",\"driver_info\":" << quoted(d.driver.driverInfo)
      << ",\"device_uuid\":" << quoted(hex(d.ids.deviceUUID)) << ",\"driver_uuid\":" << quoted(hex(d.ids.driverUUID))
      << ",\"denorm_preserve_f32\":" << (d.fp.shaderDenormPreserveFloat32 ? "true" : "false")
      << ",\"rne_f32\":" << (d.fp.shaderRoundingModeRTEFloat32 ? "true" : "false")
      << ",\"signed_zero_inf_nan_f32\":" << (d.fp.shaderSignedZeroInfNanPreserveFloat32 ? "true" : "false")
      << ",\"denorm_independence\":" << d.fp.denormBehaviorIndependence
      << ",\"rounding_independence\":" << d.fp.roundingModeIndependence << '}';
  return out.str();
}
struct Buffer { VkBuffer buffer{}; VkDeviceMemory memory{}; void *mapped = nullptr; VkDeviceSize bytes = 0; };
static Buffer allocate(Vulkan &vk, Report &r, VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes) {
  Buffer b; b.bytes = bytes;
  VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  create.size = bytes; create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  r.check(vk.CreateBuffer(device, &create, nullptr, &b.buffer), "create buffer");
  VkMemoryRequirements requirements{}; vk.GetBufferMemoryRequirements(device, b.buffer, &requirements);
  VkPhysicalDeviceMemoryProperties properties{}; vk.GetPhysicalDeviceMemoryProperties(physical, &properties);
  std::uint32_t type = UINT32_MAX;
  constexpr auto needed = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (unsigned i = 0; i < properties.memoryTypeCount; ++i)
    if ((requirements.memoryTypeBits & (1U << i)) && (properties.memoryTypes[i].propertyFlags & needed) == needed) { type = i; break; }
  if (type == UINT32_MAX) throw Refusal("no HOST_VISIBLE|HOST_COHERENT staging type for actual buffer");
  if (requirements.size < bytes) throw std::runtime_error("invalid buffer memory requirements");
  VkMemoryAllocateInfo memory{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  memory.allocationSize = requirements.size; memory.memoryTypeIndex = type;
  r.check(vk.AllocateMemory(device, &memory, nullptr, &b.memory), "allocate staging");
  r.check(vk.BindBufferMemory(device, b.buffer, b.memory, 0), "bind staging");
  r.check(vk.MapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "map coherent staging");
  return b;
}
static void release(Vulkan &vk, VkDevice device, const Buffer &b) {
  vk.UnmapMemory(device, b.memory); vk.DestroyBuffer(device, b.buffer, nullptr); vk.FreeMemory(device, b.memory, nullptr);
}
static Word read(const void *memory, std::size_t index) {
  Word value; std::memcpy(&value, static_cast<const char *>(memory) + index * 4, 4); return value;
}
static void checkAbi(const issuer::ShaderAbiV1 &abi, issuer::StaticGemmShapeV1 shape, bool fill) {
  if (abi.entry_point != issuer::kEntryPointV1 || abi.local_size != std::array<std::uint32_t,3>{1,1,1} ||
      abi.dispatch != std::array<std::uint32_t,3>{static_cast<std::uint32_t>(shape.m), static_cast<std::uint32_t>(shape.n), 1} ||
      abi.buffers.size() != (fill ? 1U : 3U)) throw std::runtime_error("unexpected checked shader ABI");
  unsigned roles = 0;
  for (const auto &b : abi.buffers) {
    const auto role = static_cast<unsigned>(b.role);
    const std::array<std::uint64_t,3> bytes{std::uint64_t(shape.m) * shape.k * 4, std::uint64_t(shape.k) * shape.n * 4, std::uint64_t(shape.m) * shape.n * 4};
    if (role > 2 || (roles & (1U << role)) || b.descriptor_set != 0 || b.binding != (fill ? 0U : role) || b.bytes != bytes[role] ||
        b.element_stride != 4 || b.member_offset != 0 || b.writable != (role == 2))
      throw std::runtime_error("unexpected checked buffer ABI");
    for (const auto &other : abi.buffers) if (&other != &b && other.binding == b.binding) throw std::runtime_error("duplicate shader binding");
    roles |= 1U << role;
  }
  if (roles != (fill ? 4U : 7U)) throw std::runtime_error("missing shader ABI role");
}
int main(int argc, char **argv) {
  Report r;
  try {
    const auto options = parse(argc, argv);
    r.field("oracle_self_controls", std::to_string(strict_oracle::selfCheck()));
    if (options.oracle_only) {
      // Small padded geometry exercises all constant and integer cross-checks.
      const auto frozen = strict_oracle::freeze(2, 7, 3);
      r.field("oracle_case_count", std::to_string(frozen.size()));
      r.field("vulkan_loaded", "false");
      return r.finish("ORACLE_SOFTWARE_ONLY", "MATCHED", "No issuance or Vulkan calls", 0);
    }
    std::vector<strict_oracle::Case> cases;
    issuer::StrictGemmArtifactV1 artifact;
    if (!options.inventory) {
      const auto s = options.shape; std::string error;
      r.field("shape_mkn", '[' + std::to_string(s.m) + ',' + std::to_string(s.k) + ',' + std::to_string(s.n) + ']');
      if (!issuer::validateStaticGemmShapeV1(s, error)) throw Refusal("issuer shape refusal: " + error);
      constexpr std::uint64_t cap = 1U << 20;
      const std::uint64_t count = 2 + (s.k >= 3 ? std::size(strict_oracle::specs) : 0), outputs = std::uint64_t(s.m) * s.n;
      if (outputs > cap / count || std::uint64_t(s.k) > (cap / count / outputs - 1) / 2)
        throw Refusal("independent total test cap 2^20 fill/multiply/add scalar operations exceeded");
      r.field("test_scalar_operations", std::to_string(count * outputs * (2 * s.k + 1)));
      r.field("test_scalar_operation_cap", std::to_string(cap));
      cases = strict_oracle::freeze(s.m, s.k, s.n);
      r.field("special_cases", s.k >= 3 ? "\"APPLICABLE_K_GE_3\"" : "\"SKIPPED_K_LT_3; base/last-K still executed\"");
      std::ostringstream skipped; skipped << '[';
      if (s.k < 3) for (std::size_t i = 0; i < std::size(strict_oracle::specs); ++i) {
        if (i) skipped << ',';
        skipped << "{\"case\":" << quoted(strict_oracle::specs[i].name) << ",\"reason\":\"requires K>=3\"}";
      }
      skipped << ']'; r.field("skipped_cases", skipped.str());
      mlir::MLIRContext context;
      artifact = issuer::issueStrictGemmArtifactV1(context, s);
      if (!artifact || !issuer::verifyStrictGemmArtifactV1(context, artifact, error)) throw std::runtime_error("issuer/replay failure: " + artifact.error + error);
      checkAbi(artifact.fill_abi, s, true); checkAbi(artifact.gemm_abi, s, false);
      r.field("artifact_replay_verified_before_vulkan", "true");
      r.field("fill_sha256", quoted(digest(artifact.fill_words))); r.field("gemm_sha256", quoted(digest(artifact.gemm_words)));
      r.field("fill_dispatch", triple(artifact.fill_abi.dispatch)); r.field("gemm_dispatch", triple(artifact.gemm_abi.dispatch));
      r.field("local_size", triple(artifact.gemm_abi.local_size));
    }
    r.field("expectations_frozen_before_vulkan_load", "true");
    Vulkan vk;
    std::uint32_t version = 0; r.check(vk.EnumerateInstanceVersion(&version), "loader version");
    r.field("loader_api_version", std::to_string(version));
    if (version < VK_API_VERSION_1_2) throw Refusal("Vulkan 1.2 loader required");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName = "mdslc-test-issuer-vulkan"; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; instanceInfo.pApplicationInfo = &app;
    VkInstance instance{}; r.check(vk.CreateInstance(&instanceInfo, nullptr, &instance), "create instance");
    std::uint32_t count = 0; r.check(vk.EnumeratePhysicalDevices(instance, &count, nullptr), "device count");
    std::vector<VkPhysicalDevice> devices(count);
    if (count) r.check(vk.EnumeratePhysicalDevices(instance, &count, devices.data()), "devices");
    std::ostringstream inventory; inventory << '[';
    for (unsigned i = 0; i < count; ++i) { if (i) inventory << ','; inventory << deviceJson(i, info(vk, devices[i])); }
    inventory << ']'; r.field("devices", inventory.str());
    if (options.inventory) { vk.DestroyInstance(instance, nullptr); return r.finish("INVENTORY", "NO_DISPATCH", "", 0); }
    if (options.device >= count) throw Refusal("explicit device index unavailable; no fallback");
    const auto physical = devices[options.device]; const auto d = info(vk, physical);
    r.field("selected_device_index", std::to_string(options.device));
    if (d.properties.apiVersion < VK_API_VERSION_1_2 || !d.fp.shaderDenormPreserveFloat32 || !d.fp.shaderRoundingModeRTEFloat32 || !d.fp.shaderSignedZeroInfNanPreserveFloat32)
      throw Refusal("selected device lacks Vulkan1.2/DenormPreserve/RTE/SignedZeroInfNan f32 controls");
    if (d.properties.vendorID != 0x1002 || d.driver.driverID != VK_DRIVER_ID_MESA_RADV ||
        (d.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && d.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU))
      throw Refusal("positive sample requires selected physical AMD RADV device");
    const auto &limits = d.properties.limits;
    if (limits.maxBoundDescriptorSets < 1 || limits.maxPerStageDescriptorStorageBuffers < 3 || limits.maxDescriptorSetStorageBuffers < 3 || limits.maxPerStageResources < 3 || limits.maxMemoryAllocationCount < 3)
      throw Refusal("actual descriptor/allocation limits insufficient");
    for (const auto *abi : {&artifact.fill_abi, &artifact.gemm_abi}) {
      for (unsigned axis = 0; axis < 3; ++axis)
        if (abi->dispatch[axis] > limits.maxComputeWorkGroupCount[axis] || abi->local_size[axis] > limits.maxComputeWorkGroupSize[axis]) throw Refusal("actual dispatch/local-size limit");
      if (limits.maxComputeWorkGroupInvocations < 1) throw Refusal("actual invocation limit");
      for (const auto &b : abi->buffers) if (b.bytes > limits.maxStorageBufferRange) throw Refusal("actual storage-buffer range limit");
    }
    const std::uint64_t alignment = std::max<std::uint64_t>(4, limits.minStorageBufferOffsetAlignment);
    if (alignment > 65536 || alignment % 4) throw Refusal("test staging alignment envelope exceeded");
    const std::uint64_t offset = ((256 + alignment - 1) / alignment) * alignment;
    r.field("output_descriptor_offset", std::to_string(offset)); r.field("min_storage_buffer_offset_alignment", std::to_string(limits.minStorageBufferOffsetAlignment));
    r.field("max_storage_buffer_range", std::to_string(limits.maxStorageBufferRange));
    r.field("max_dispatch", triple({limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1], limits.maxComputeWorkGroupCount[2]}));
    std::uint32_t familyCount = 0; vk.GetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount); vk.GetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    std::uint32_t family = UINT32_MAX;
    for (unsigned i = 0; i < familyCount; ++i) if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    if (family == UINT32_MAX) throw Refusal("selected device has no compute queue");
    const float priority = 1;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; queueInfo.queueFamilyIndex = family; queueInfo.queueCount = 1; queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; deviceInfo.queueCreateInfoCount = 1; deviceInfo.pQueueCreateInfos = &queueInfo;
    VkDevice device{}; r.check(vk.CreateDevice(physical, &deviceInfo, nullptr, &device), "create selected device");
    VkQueue queue{}; vk.GetDeviceQueue(device, family, 0, &queue);
    std::array<VkShaderModule,2> modules{}; std::array<VkDescriptorSetLayout,2> layouts{};
    std::array<VkPipelineLayout,2> pipelineLayouts{}; std::array<VkPipeline,2> pipelines{};
    const std::array<const issuer::ShaderAbiV1 *,2> abis{&artifact.fill_abi, &artifact.gemm_abi};
    for (unsigned stage = 0; stage < 2; ++stage) {
      const auto &words = stage ? artifact.gemm_words : artifact.fill_words;
      VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; shader.codeSize = words.size() * 4; shader.pCode = words.data();
      r.check(vk.CreateShaderModule(device, &shader, nullptr, &modules[stage]), "create directly issued/replayed shader");
      std::vector<VkDescriptorSetLayoutBinding> bindings;
      for (const auto &b : abis[stage]->buffers) bindings.push_back({b.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
      VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; layout.bindingCount = bindings.size(); layout.pBindings = bindings.data();
      r.check(vk.CreateDescriptorSetLayout(device, &layout, nullptr, &layouts[stage]), "descriptor layout");
      VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pipelineLayout.setLayoutCount = 1; pipelineLayout.pSetLayouts = &layouts[stage];
      r.check(vk.CreatePipelineLayout(device, &pipelineLayout, nullptr, &pipelineLayouts[stage]), "pipeline layout");
      VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pipeline.stage.module = modules[stage]; pipeline.stage.pName = abis[stage]->entry_point.c_str(); pipeline.layout = pipelineLayouts[stage];
      r.check(vk.CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipelines[stage]), "pipeline");
    }
    const auto s = options.shape;
    std::array<Buffer,3> buffers{allocate(vk,r,physical,device,std::uint64_t(s.m)*s.k*4), allocate(vk,r,physical,device,std::uint64_t(s.k)*s.n*4), allocate(vk,r,physical,device,offset+std::uint64_t(s.m)*s.n*4+256)};
    std::array<VkDescriptorBufferInfo,3> descriptors{{{buffers[0].buffer,0,std::uint64_t(s.m)*s.k*4},{buffers[1].buffer,0,std::uint64_t(s.k)*s.n*4},{buffers[2].buffer,offset,std::uint64_t(s.m)*s.n*4}}};
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; poolInfo.maxSets=2; poolInfo.poolSizeCount=1; poolInfo.pPoolSizes=&poolSize;
    VkDescriptorPool pool{}; r.check(vk.CreateDescriptorPool(device,&poolInfo,nullptr,&pool),"descriptor pool");
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; setInfo.descriptorPool=pool; setInfo.descriptorSetCount=2; setInfo.pSetLayouts=layouts.data();
    std::array<VkDescriptorSet,2> sets{}; r.check(vk.AllocateDescriptorSets(device,&setInfo,sets.data()),"descriptor sets");
    for(unsigned stage=0;stage<2;++stage) for(const auto &b:abis[stage]->buffers) {
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=sets[stage]; write.dstBinding=b.binding; write.descriptorCount=1; write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&descriptors[static_cast<unsigned>(b.role)];
      vk.UpdateDescriptorSets(device,1,&write,0,nullptr);
    }
    VkCommandPoolCreateInfo commandPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; commandPoolInfo.queueFamilyIndex=family;
    VkCommandPool commandPool{}; r.check(vk.CreateCommandPool(device,&commandPoolInfo,nullptr,&commandPool),"command pool");
    VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; commandInfo.commandPool=commandPool; commandInfo.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; commandInfo.commandBufferCount=1;
    VkCommandBuffer command{}; r.check(vk.AllocateCommandBuffers(device,&commandInfo,&command),"command buffer");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence{}; r.check(vk.CreateFence(device,&fenceInfo,nullptr,&fence),"fence");
    std::ostringstream results; results << '[';
    for(std::size_t index=0;index<cases.size();++index) {
      const auto &c=cases[index];
      std::memcpy(buffers[0].mapped,c.a.data(),c.a.size()*4); std::memcpy(buffers[1].mapped,c.b.data(),c.b.size()*4);
      constexpr Word sentinel=0x7fc0cafe;
      for(std::size_t p=0;p<buffers[2].bytes/4;++p) std::memcpy(static_cast<char*>(buffers[2].mapped)+p*4,&sentinel,4);
      // A NaN sentinel cannot witness a missing write when NaN class is expected.
      // Use a finite nonzero sentinel for exactly those output elements instead.
      for(std::size_t p=0;p<c.expected.size();++p) if(strict_oracle::nan(c.expected[p])) {
        constexpr Word finite=0x42f60000;
        std::memcpy(static_cast<char*>(buffers[2].mapped)+offset+p*4,&finite,4);
      }
      r.check(vk.ResetCommandPool(device,commandPool,0),"reset completed commands"); r.check(vk.ResetFences(device,1,&fence),"reset fence");
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; r.check(vk.BeginCommandBuffer(command,&begin),"begin commands");
      std::array<VkBufferMemoryBarrier,3> barriers{};
      for(unsigned i=0;i<3;++i) { auto &b=barriers[i]; b.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; b.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=buffers[i].buffer; b.size=VK_WHOLE_SIZE; }
      vk.CmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,barriers.size(),barriers.data(),0,nullptr);
      for(unsigned stage=0;stage<2;++stage) {
        if(stage) { auto b=barriers[2]; b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; b.offset=offset; b.size=descriptors[2].range; vk.CmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&b,0,nullptr); }
        vk.CmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[stage]); vk.CmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayouts[stage],0,1,&sets[stage],0,nullptr);
        const auto &dispatch=abis[stage]->dispatch; vk.CmdDispatch(command,dispatch[0],dispatch[1],dispatch[2]);
      }
      for(auto &b:barriers) { b.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_HOST_READ_BIT; }
      vk.CmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,barriers.size(),barriers.data(),0,nullptr);
      r.check(vk.EndCommandBuffer(command),"end commands");
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&command;
      r.check(vk.QueueSubmit(queue,1,&submit,fence),"submit; completion unknown on failure"); r.dispatches+=2;
      r.check(vk.WaitForFences(device,1,&fence,VK_TRUE,5000000000ULL),"wait; completion unknown on failure"); ++r.completions;
      // HOST_COHERENT plus the memory dependency and fence: no invalidate needed.
      std::uint64_t bad=0;
      for(unsigned input=0;input<2;++input) { const auto &expected=input?c.b:c.a; for(std::size_t p=0;p<expected.size();++p) { ++r.inputs; bad+=read(buffers[input].mapped,p)!=expected[p]; } }
      for(std::size_t p=0;p<buffers[2].bytes/4;++p) if(p<offset/4 || p>=offset/4+c.expected.size()) { ++r.canaries; bad+=read(buffers[2].mapped,p)!=sentinel; }
      std::string first="null";
      for(std::size_t p=0;p<c.expected.size();++p) { ++r.outputs; const Word actual=read(buffers[2].mapped,offset/4+p); if(!strict_oracle::equal(c.expected[p],actual)) { ++bad; if(first=="null") first="{\"index\":"+std::to_string(p)+",\"expected_bits\":"+std::to_string(c.expected[p])+",\"actual_bits\":"+std::to_string(actual)+'}'; } }
      r.mismatches+=bad;
      if(index) results<<',';
      results<<"{\"case\":"<<quoted(c.name)<<",\"outputs\":"<<c.expected.size()<<",\"expected_sha256\":"<<quoted(digest(c.expected))<<",\"mismatches\":"<<bad<<",\"first_output_mismatch\":"<<first<<'}';
    }
    results<<']'; r.field("cases",results.str()); r.field("staging","\"HOST_VISIBLE|HOST_COHERENT only; three disjoint allocations\"");
    r.field("visibility","\"HOST->COMPUTE; fill->GEMM; COMPUTE->HOST; fenced completion before inspection/free\"");
    for(const auto &b:buffers) release(vk,device,b);
    vk.DestroyFence(device,fence,nullptr); vk.DestroyCommandPool(device,commandPool,nullptr); vk.DestroyDescriptorPool(device,pool,nullptr);
    for(unsigned i=0;i<2;++i) { vk.DestroyPipeline(device,pipelines[i],nullptr); vk.DestroyPipelineLayout(device,pipelineLayouts[i],nullptr); vk.DestroyDescriptorSetLayout(device,layouts[i],nullptr); vk.DestroyShaderModule(device,modules[i],nullptr); }
    vk.DestroyDevice(device,nullptr); vk.DestroyInstance(instance,nullptr);
    return r.finish(r.mismatches?"COUNTEREXAMPLE":"BOUNDED_PHYSICAL_SAMPLE",r.mismatches?"MISMATCH":"MATCHED","",r.mismatches?1:0);
  } catch(const Refusal &error) { return r.finish("REFUSED","NO_FALLBACK",error.what(),78); }
    catch(const std::exception &error) { return r.finish("ERROR","PROBE_ERROR",error.what(),1); }
}
