// Isolated research host: no Matcore linking, candidate, source, or publication.
// Only the separately derived/validated static fill+GEMM SPIR-V is exercised.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>

#pragma STDC FENV_ACCESS ON

static std::string quoted(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << char(c);
  }
  out << '"';
  return out.str();
}

struct Report {
  std::string path;
  std::ostringstream fields;
  void field(const std::string &key, const std::string &json) {
    fields << ",\n" << quoted(key) << ':' << json;
  }
  bool save(const std::string &status, const std::string &error = "") {
    std::ostringstream out;
    out << "{\n\"classification\":\"bounded static Vulkan research; no MDSLC target/source/runtime authority\","
           "\n\"strict_contract_qualified\":false,\n\"status\":" << quoted(status)
        << ",\n\"error\":" << quoted(error) << fields.str() << "\n}\n";
    std::ofstream file(path);
    file << out.str();
    file.close();
    if (!file) return false;
    std::fputs(out.str().c_str(), stdout);
    std::fflush(stdout);
    return true;
  }
  [[noreturn]] void fail(const std::string &phase, VkResult result = VK_SUCCESS) {
    save("PROBE_ERROR", phase + "; VkResult=" + std::to_string(result));
    // Deliberate research fail-stop, not a reusable runtime quarantine policy.
    // On unknown completion, never destroy/free/unmap possibly live objects.
    // No caller-owned output or publication exists in this isolated process.
    std::_Exit(1);
  }
  void check(VkResult result, const std::string &phase) {
    if (result != VK_SUCCESS) fail(phase, result);
  }
};

// Bounded artifact control check, NOT an equivalence proof or executable issuer.
// Numeric opcodes come from the SPIR-V 1.3 grammar/float-controls extension.
static std::vector<std::uint32_t> readControlled(const char *path, bool gemm) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  auto size = input.tellg();
  if (!input || size < 20 || size > 65536 || size % 4) throw std::runtime_error("invalid SPIR-V byte extent");
  std::vector<std::uint32_t> words(std::size_t(size) / 4);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(words.data()), size);
  if (!input || words[0] != 0x07230203 || words[1] != 0x00010300 || words[4] != 0)
    throw std::runtime_error("expected known SPIR-V 1.3 binary");
  std::set<std::uint32_t> capabilities, noContract, arithmetic, bindings;
  std::vector<std::array<std::uint32_t, 3>> modes;
  unsigned entries = 0, multiply = 0, add = 0;
  std::uint32_t entry = 0;
  bool extension = false;
  std::vector<std::uint32_t> localSizeEntries;
  for (std::size_t i = 5; i < words.size();) {
    unsigned count = words[i] >> 16, opcode = words[i] & 65535;
    if (!count || count > words.size() - i) throw std::runtime_error("malformed instruction extent");
    const auto *v = words.data() + i;
    auto literal = [&](unsigned start) {
      if (start >= count) throw std::runtime_error("missing literal string");
      const char *p = reinterpret_cast<const char *>(v + start);
      std::size_t bytes = (count - start) * 4;
      const char *end = static_cast<const char *>(std::memchr(p, 0, bytes));
      if (!end) throw std::runtime_error("unterminated literal");
      return std::string(p, end);
    };
    if (opcode == 17) { // OpCapability
      if (count != 2 || !capabilities.insert(v[1]).second) throw std::runtime_error("duplicate capability");
    } else if (opcode == 10) { // OpExtension
      if (literal(1) == "SPV_KHR_float_controls") extension = true;
    } else if (opcode == 15) { // OpEntryPoint
      if (count < 4 || v[1] != 5 || literal(3) != "spirv_static_specimen_kernel")
        throw std::runtime_error("unexpected compute entry");
      ++entries;
      entry = v[2];
    } else if (opcode == 16) { // OpExecutionMode
      if (count == 6 && v[2] == 17 && v[3] == 1 && v[4] == 1 && v[5] == 1) localSizeEntries.push_back(v[1]);
      else if (count == 4) modes.push_back({v[1], v[2], v[3]});
      else throw std::runtime_error("unexpected execution mode");
    } else if (opcode == 71 || opcode == 72) { // OpDecorate/OpMemberDecorate
      unsigned d = opcode == 71 ? 2 : 3;
      if (count <= d) throw std::runtime_error("malformed decoration");
      if (v[d] == 0 || v[d] == 39 || v[d] == 40) throw std::runtime_error("forbidden relaxed/rounding/fast-math decoration");
      if (opcode == 71 && v[2] == 42) {
        if (count != 3 || !noContract.insert(v[1]).second) throw std::runtime_error("invalid NoContraction decoration");
      }
      if (opcode == 71 && v[2] == 33) {
        if (count != 4 || !bindings.insert(v[3]).second) throw std::runtime_error("unexpected duplicate binding");
      }
      if (opcode == 71 && v[2] == 34 && (count != 4 || v[3] != 0)) throw std::runtime_error("unexpected descriptor set");
    } else if (opcode == 129 || opcode == 133) { // OpFAdd/OpFMul
      if (count != 5 || !arithmetic.insert(v[2]).second) throw std::runtime_error("unexpected arithmetic result");
      opcode == 129 ? ++add : ++multiply;
    } else if (opcode == 12 || opcode == 57) {
      throw std::runtime_error("unexpected extended instruction/function call");
    } else if (opcode == 22 && (count != 3 || v[2] != 32)) {
      throw std::runtime_error("non-f32 type");
    }
    i += count;
  }
  const std::set<std::uint32_t> wantedCapabilities{1, 4464, 4466, 4467};
  const std::set<std::uint32_t> wantedModes{4459, 4461, 4462};
  std::set<std::uint32_t> foundModes;
  for (auto mode : modes)
    if (mode[0] != entry || mode[2] != 32 || !foundModes.insert(mode[1]).second)
      throw std::runtime_error("invalid/duplicate f32 mode");
  if (entries != 1 || !extension || localSizeEntries != std::vector<std::uint32_t>{entry} || capabilities != wantedCapabilities ||
      foundModes != wantedModes || noContract != arithmetic || multiply != unsigned(gemm) || add != unsigned(gemm) ||
      bindings != (gemm ? std::set<std::uint32_t>{0, 1, 2} : std::set<std::uint32_t>{0}))
    throw std::runtime_error("missing required strict controls or known interface");
  return words;
}

struct Case {
  const char *name;
  std::array<float, 6> a;
  std::array<float, 12> b;
};
static float value(std::uint32_t u) { return std::bit_cast<float>(u); }
static std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }
static std::uint32_t mappedWord(const void *memory, unsigned index) {
  std::uint32_t result;
  std::memcpy(&result, static_cast<const char *>(memory) + 4 * index, 4);
  return result;
}
static float reference(const Case &c, unsigned i, unsigned j, bool reverse = false) {
  volatile float sum = 0.0f;
  for (unsigned n = 0; n < 3; ++n) {
    unsigned k = reverse ? 2 - n : n;
    volatile float product = c.a[i * 3 + k] * c.b[k * 4 + j];
    sum = sum + product;
  }
  return sum;
}
static Case dotCase(const char *name, std::array<float, 3> a, std::array<float, 3> b) {
  Case c{name, {}, {}};
  for (unsigned i = 0; i < 2; ++i) for (unsigned k = 0; k < 3; ++k) c.a[i * 3 + k] = a[k];
  for (unsigned k = 0; k < 3; ++k) for (unsigned j = 0; j < 4; ++j) c.b[k * 4 + j] = b[k];
  return c;
}
struct HostFp {
  fenv_t saved{};
  unsigned mxcsr = _mm_getcsr();
  HostFp() {
    if (fegetenv(&saved) || fesetround(FE_TONEAREST)) throw std::runtime_error("host FP setup failed");
    _mm_setcsr(_mm_getcsr() & ~unsigned((1 << 15) | (1 << 6)));
    if (fegetround() != FE_TONEAREST || (_mm_getcsr() & ((1 << 15) | (1 << 6)))) throw std::runtime_error("invalid host oracle FP controls");
  }
  ~HostFp() { fesetenv(&saved); _mm_setcsr(mxcsr); }
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  bool coherent = false;
};
static Buffer allocate(Report &r, VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes) {
  Buffer b;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = bytes;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  r.check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "create storage buffer");
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(device, b.buffer, &requirements);
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physical, &properties);
  std::uint32_t selected = UINT32_MAX;
  for (unsigned i = 0; i < properties.memoryTypeCount; ++i)
    if ((requirements.memoryTypeBits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      selected = i;
      if (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) break;
    }
  if (selected == UINT32_MAX) r.fail("no host-visible staging memory");
  b.coherent = properties.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = selected;
  r.check(vkAllocateMemory(device, &allocation, nullptr, &b.memory), "allocate explicit staging");
  r.check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "bind staging");
  r.check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "map staging");
  return b;
}
static void hostVisibility(Report &r, VkDevice device, const Buffer &b, bool invalidate) {
  if (b.coherent) return;
  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  range.memory = b.memory;
  // Descriptor offsets are NOT cache-maintenance offsets. The entire mapped
  // allocation at offset0 + VK_WHOLE_SIZE respects nonCoherentAtomSize.
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  r.check(invalidate ? vkInvalidateMappedMemoryRanges(device, 1, &range) : vkFlushMappedMemoryRanges(device, 1, &range),
          invalidate ? "invalidate host staging" : "flush host staging");
}
static void release(VkDevice device, const Buffer &b) {
  vkUnmapMemory(device, b.memory);
  vkDestroyBuffer(device, b.buffer, nullptr);
  vkFreeMemory(device, b.memory, nullptr);
}

int main(int argc, char **argv) {
  if (argc != 5) { std::fputs("usage: probe fill-strict.spv gemm-strict.spv {inventory|device-index} evidence.json\n", stderr); return 2; }
  Report r{argv[4]};
  try {
    bool inventoryOnly = std::strcmp(argv[3], "inventory") == 0;
    unsigned chosen = UINT32_MAX;
    if (!inventoryOnly) {
      auto parsed = std::from_chars(argv[3], argv[3] + std::strlen(argv[3]), chosen);
      if (parsed.ec != std::errc{} || *parsed.ptr) throw std::runtime_error("invalid explicit device index");
    }
    const auto fillWords = readControlled(argv[1], false), gemmWords = readControlled(argv[2], true);
    r.field("binary_control_checks", "true");
    HostFp fp;
    std::vector<Case> cases{{"rectangular", {1, -2, 3, -4, 5, 6}, {2, 3, -4, 5, 6, 7, 8, -9, 10, -11, 12, 13}}};
    cases.push_back(dotCase("subnormal-input", {value(1), 0, 0}, {1, 1, 1}));
    cases.push_back(dotCase("subnormal-result", {value(0x00800000), 0, 0}, {0.5f, 1, 1}));
    cases.push_back(dotCase("negative-subnormal", {value(0x80000001), 0, 0}, {1, 1, 1}));
    cases.push_back(dotCase("signed-zero", {-0.0f, -0.0f, -0.0f}, {1, 1, 1}));
    cases.push_back(dotCase("fma-discriminator", {-1, value(0x3f800001), 0}, {1, value(0x3f7ffffe), 1}));
    cases.push_back(dotCase("increasing-k", {16777216.0f, 1, -16777216.0f}, {1, 1, 1}));
    cases.push_back(dotCase("rte-not-rtz", {1, value(0x33c00000), 0}, {1, 1, 1}));
    cases.push_back(dotCase("infinity", {INFINITY, 1, 1}, {1, 1, 1}));
    cases.push_back(dotCase("quiet-nan", {value(0x7fc00001), 1, 1}, {1, 1, 1}));
    if (bits(reference(cases[1], 0, 0)) != 1 || bits(reference(cases[2], 0, 0)) != 0x00400000 ||
        bits(reference(cases[3], 0, 0)) != 0x80000001 || bits(reference(cases[4], 0, 0)) != 0 ||
        bits(reference(cases[5], 0, 0)) != 0 || std::fma(value(0x3f800001), value(0x3f7ffffe), -1.0f) == 0 ||
        reference(cases[6], 0, 0) != 0 || reference(cases[6], 0, 0, true) != 1 ||
        bits(reference(cases[7], 0, 0)) != 0x3f800001) throw std::runtime_error("host adversarial oracle failed");
    r.field("host_oracle_controls", "9");
    r.field("order_witness", "\"[2^24,1,-2^24]: forward0 reverse1\"");
    // Freeze every expected bit before even loading/enumerating Vulkan devices.
    // A driver changing host FP controls cannot bias the comparison oracle.
    std::vector<std::array<std::uint32_t, 8>> expectedBits(cases.size());
    for (unsigned c = 0; c < cases.size(); ++c)
      for (unsigned i = 0; i < 2; ++i) for (unsigned j = 0; j < 4; ++j)
        expectedBits[c][i * 4 + j] = bits(reference(cases[c], i, j));
    const unsigned hostControls = _mm_getcsr() & 0xffc0u;
    r.field("expectations_frozen_before_vulkan", "true");
    std::uint32_t loader = 0;
    r.check(vkEnumerateInstanceVersion(&loader), "query Vulkan loader");
    if (loader < VK_API_VERSION_1_2) { r.save("REFUSED_VULKAN_VERSION", "Vulkan1.2 core float-controls required"); return 78; }
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "mdslc-static-spirv-research";
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &application;
    VkInstance instance;
    r.check(vkCreateInstance(&instanceInfo, nullptr, &instance), "create Vulkan1.2 instance");
    std::uint32_t count = 0;
    r.check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate physical devices");
    if (!count || (!inventoryOnly && chosen >= count)) { r.save("REFUSED_DEVICE_INDEX", "no selected physical device"); vkDestroyInstance(instance, nullptr); return 78; }
    std::vector<VkPhysicalDevice> devices(count);
    r.check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "read physical devices");
    std::ostringstream inventory;
    inventory << '[';
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceFloatControlsProperties controls{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    for (unsigned i = 0; i < count; ++i) {
      VkPhysicalDeviceFloatControlsProperties current{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
      VkPhysicalDeviceDriverProperties currentDriver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      current.pNext = &currentDriver;
      VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      p.pNext = &current;
      vkGetPhysicalDeviceProperties2(devices[i], &p);
      if (i) inventory << ',';
      inventory << "{\"index\":" << i << ",\"name\":" << quoted(p.properties.deviceName)
                << ",\"vendor\":" << p.properties.vendorID << ",\"device_id\":" << p.properties.deviceID
                << ",\"api\":" << p.properties.apiVersion << ",\"driver_id\":" << currentDriver.driverID
                << ",\"driver_name\":" << quoted(currentDriver.driverName) << ",\"driver_info\":" << quoted(currentDriver.driverInfo)
                << ",\"denorm_preserve_f32\":" << (current.shaderDenormPreserveFloat32 ? "true" : "false")
                << ",\"rte_f32\":" << (current.shaderRoundingModeRTEFloat32 ? "true" : "false")
                << ",\"signed_zero_inf_nan_f32\":" << (current.shaderSignedZeroInfNanPreserveFloat32 ? "true" : "false") << '}';
      if (i == chosen) { properties = p.properties; controls = current; driver = currentDriver; }
    }
    inventory << ']';
    r.field("devices", inventory.str());
    r.field("selected_device_index", inventoryOnly ? "null" : std::to_string(chosen));
    if (inventoryOnly) {
      r.field("dispatches", "0");
      bool saved = r.save("INVENTORIED_NO_DISPATCH");
      vkDestroyInstance(instance, nullptr);
      return saved ? 0 : 1;
    }
    if (properties.apiVersion < VK_API_VERSION_1_2 || !controls.shaderDenormPreserveFloat32 ||
        !controls.shaderRoundingModeRTEFloat32 || !controls.shaderSignedZeroInfNanPreserveFloat32) {
      r.field("dispatches", "0");
      r.save("REFUSED_FLOAT_CONTROLS", "selected device lacks required Vulkan1.2/f32 controls; no fallback");
      vkDestroyInstance(instance, nullptr);
      return 78;
    }
    if (properties.vendorID != 0x1002 || driver.driverID != VK_DRIVER_ID_MESA_RADV ||
        (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
      r.field("dispatches", "0");
      r.save("REFUSED_DEVICE_IDENTITY", "this research lane requires a physical AMD RADV device");
      vkDestroyInstance(instance, nullptr);
      return 78;
    }
    r.field("vulkan_feature_enablement", "\"Vulkan1.2 core shader_float_controls; no optional feature/extension enabled or required for this scalar-f32 module\"");
    const auto &limits = properties.limits;
    VkDeviceSize alignment = limits.minStorageBufferOffsetAlignment;
    if (!alignment || alignment > 65536 || limits.maxStorageBufferRange < 48 ||
        limits.maxComputeWorkGroupCount[0] < 2 || limits.maxComputeWorkGroupCount[1] < 4) throw std::runtime_error("static resource limits not satisfied");
    VkDeviceSize offset = ((256 + alignment - 1) / alignment) * alignment;
    if (offset % 4) throw std::runtime_error("non-f32 descriptor offset");
    VkDeviceSize outputBytes = offset + 32 + 256;
    r.field("output_descriptor_offset", std::to_string(offset));
    r.field("noncoherent_atom_size", std::to_string(limits.nonCoherentAtomSize));
    std::uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devices[chosen], &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(devices[chosen], &familyCount, families.data());
    unsigned family = UINT32_MAX;
    for (unsigned i = 0; i < familyCount; ++i) if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    if (family == UINT32_MAX) throw std::runtime_error("no compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = family;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled{}; // No hidden optional features.
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    VkDevice device;
    r.check(vkCreateDevice(devices[chosen], &deviceInfo, nullptr, &device), "create explicitly gated device");
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    auto shader = [&](const std::vector<std::uint32_t> &words) {
      VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      info.codeSize = words.size() * 4;
      info.pCode = words.data();
      VkShaderModule module;
      r.check(vkCreateShaderModule(device, &info, nullptr, &module), "create validated shader module");
      return module;
    };
    VkShaderModule fill = shader(fillWords), gemm = shader(gemmWords);
    auto setLayout = [&](unsigned n) {
      std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
      for (unsigned i = 0; i < n; ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      info.bindingCount = n;
      info.pBindings = bindings.data();
      VkDescriptorSetLayout layout;
      r.check(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout), "create known descriptor layout");
      return layout;
    };
    std::array<VkDescriptorSetLayout, 2> layouts{setLayout(1), setLayout(3)};
    std::array<VkPipelineLayout, 2> pipelineLayouts{};
    std::array<VkPipeline, 2> pipelines{};
    for (unsigned i = 0; i < 2; ++i) {
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.setLayoutCount = 1;
      layout.pSetLayouts = &layouts[i];
      r.check(vkCreatePipelineLayout(device, &layout, nullptr, &pipelineLayouts[i]), "create pipeline layout");
      VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      info.stage.module = i ? gemm : fill;
      info.stage.pName = "spirv_static_specimen_kernel";
      info.layout = pipelineLayouts[i];
      r.check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipelines[i]), "compile strict SPIR-V pipeline");
    }
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &size;
    VkDescriptorPool pool;
    r.check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool), "create descriptor pool");
    VkDescriptorSetAllocateInfo setsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setsInfo.descriptorPool = pool;
    setsInfo.descriptorSetCount = 2;
    setsInfo.pSetLayouts = layouts.data();
    std::array<VkDescriptorSet, 2> sets{};
    r.check(vkAllocateDescriptorSets(device, &setsInfo, sets.data()), "allocate known descriptor sets");
    VkCommandPoolCreateInfo commandPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPoolInfo.queueFamilyIndex = family;
    VkCommandPool commandPool;
    r.check(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool), "create command pool");
    VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandInfo.commandPool = commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    VkCommandBuffer command;
    r.check(vkAllocateCommandBuffers(device, &commandInfo, &command), "allocate command buffer");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    r.check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "create completion fence");
    unsigned comparisons = 0, mismatches = 0, canaries = 0, immutable = 0;
    unsigned coherent = 0, noncoherent = 0;
    std::ostringstream results;
    results << '[';
    for (unsigned caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
      const auto &c = cases[caseIndex];
      Buffer a = allocate(r, devices[chosen], device, sizeof(c.a));
      Buffer b = allocate(r, devices[chosen], device, sizeof(c.b));
      Buffer e = allocate(r, devices[chosen], device, outputBytes);
      for (const auto &buffer : {a, b, e}) buffer.coherent ? ++coherent : ++noncoherent;
      std::memcpy(a.mapped, c.a.data(), sizeof(c.a));
      std::memcpy(b.mapped, c.b.data(), sizeof(c.b));
      auto *destination = static_cast<std::uint32_t *>(e.mapped);
      std::fill_n(destination, std::size_t(outputBytes / 4), 0x42f60000u);
      for (const auto &buffer : {a, b, e}) hostVisibility(r, device, buffer, false);
      std::array<VkDescriptorBufferInfo, 3> descriptors{{{a.buffer, 0, sizeof(c.a)}, {b.buffer, 0, sizeof(c.b)}, {e.buffer, offset, 32}}};
      std::array<VkWriteDescriptorSet, 4> writes{};
      for (unsigned i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = sets[i ? 1 : 0];
        writes[i].dstBinding = i ? i - 1 : 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &descriptors[i ? i - 1 : 2];
      }
      vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
      r.check(vkResetCommandPool(device, commandPool, 0), "reset completed command pool");
      r.check(vkResetFences(device, 1, &fence), "reset completed fence");
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      r.check(vkBeginCommandBuffer(command, &begin), "begin commands");
      std::array<VkBufferMemoryBarrier, 3> barriers{};
      for (unsigned i = 0; i < 3; ++i) {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].buffer = descriptors[i].buffer;
        barriers[i].size = VK_WHOLE_SIZE;
      }
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, barriers.size(), barriers.data(), 0, nullptr);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[0]);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayouts[0], 0, 1, &sets[0], 0, nullptr);
      vkCmdDispatch(command, 2, 4, 1);
      auto between = barriers[2];
      between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      between.offset = offset;
      between.size = 32;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &between, 0, nullptr);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[1]);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayouts[1], 0, 1, &sets[1], 0, nullptr);
      vkCmdDispatch(command, 2, 4, 1);
      for (auto &barrier : barriers) { barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT; }
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, barriers.size(), barriers.data(), 0, nullptr);
      r.check(vkEndCommandBuffer(command), "end commands");
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command;
      // Any failed submit/wait takes fail-stop without freeing possibly live data.
      r.check(vkQueueSubmit(queue, 1, &submit, fence), "submit strict commands");
      r.check(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull), "checked GPU completion");
      for (const auto &buffer : {a, b, e}) hostVisibility(r, device, buffer, true);
      for (std::size_t i = 0; i < outputBytes / 4; ++i)
        if (i < offset / 4 || i >= offset / 4 + 8) { ++canaries; if (destination[i] != 0x42f60000) r.fail("output guard overwritten"); }
      for (unsigned i = 0; i < c.a.size(); ++i) { ++immutable; if (mappedWord(a.mapped, i) != bits(c.a[i])) r.fail("A modified"); }
      for (unsigned i = 0; i < c.b.size(); ++i) { ++immutable; if (mappedWord(b.mapped, i) != bits(c.b[i])) r.fail("B modified"); }
      if (caseIndex) results << ',';
      results << "{\"case\":" << quoted(c.name) << ",\"outputs\":[";
      for (unsigned i = 0; i < 2; ++i) for (unsigned j = 0; j < 4; ++j) {
        unsigned n = i * 4 + j;
        std::uint32_t expected = expectedBits[caseIndex][n], actual = destination[offset / 4 + n];
        auto isNan = [](std::uint32_t u) { return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu); };
        bool equal = isNan(expected) ? isNan(actual) : expected == actual;
        ++comparisons;
        mismatches += !equal;
        if (n) results << ',';
        results << "{\"expected_bits\":" << expected << ",\"actual_bits\":" << actual << ",\"equal\":" << (equal ? "true" : "false") << '}';
      }
      results << "]}";
      // Only after established fence completion and visibility may objects retire.
      release(device, a); release(device, b); release(device, e);
    }
    results << ']';
    r.field("results", results.str());
    r.field("comparisons", std::to_string(comparisons));
    r.field("mismatches", std::to_string(mismatches));
    r.field("canary_checks", std::to_string(canaries));
    r.field("immutable_input_checks", std::to_string(immutable));
    r.field("host_coherent_allocations", std::to_string(coherent));
    r.field("host_noncoherent_allocations", std::to_string(noncoherent));
    r.field("dispatches", "20");
    r.field("checked_completions", "10");
    r.field("bounded_strict_shader_sample_matched", mismatches ? "false" : "true");
    r.field("visibility", "\"explicit HOST-to-COMPUTE, fill-to-GEMM, COMPUTE-to-HOST barriers; whole-allocation flush/invalidate if noncoherent\"");
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    for (unsigned i = 0; i < 2; ++i) { vkDestroyPipeline(device, pipelines[i], nullptr); vkDestroyPipelineLayout(device, pipelineLayouts[i], nullptr); vkDestroyDescriptorSetLayout(device, layouts[i], nullptr); }
    vkDestroyShaderModule(device, fill, nullptr); vkDestroyShaderModule(device, gemm, nullptr);
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    if (fegetround() != FE_TONEAREST || (_mm_getcsr() & 0xffc0u) != hostControls) r.fail("Vulkan changed host FP controls");
    r.field("host_fp_controls_unchanged", "true");
    return r.save(mismatches ? "EXECUTED_STRICT_COUNTEREXAMPLE" : "EXECUTED_BOUNDED_STRICT_SAMPLE_MATCH") ? 0 : 1;
  } catch (const std::exception &error) {
    return r.save("PROBE_ERROR", error.what()) ? 1 : 2;
  }
}
