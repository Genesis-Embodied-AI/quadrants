// Read-only observations of the Metal device borrowed from the pinned MoltenVK library.
// Included after the forwarding recorder's original() helper. Does not acquire a new Metal device or retain resources.
#include <array>
#include <unordered_map>
#include <vector>

namespace metal_stats {
struct Snapshot {
  U64 allocated;
  U64 recommended;
};
static std::mutex device_mutex;
static std::unordered_map<Ptr, Ptr> devices;

static Ptr selector(const char *name) {
  static auto function = reinterpret_cast<Ptr (*)(const char *)>(dlsym(RTLD_DEFAULT, "sel_registerName"));
  if (!function)
    std::abort();
  return function(name);
}

static Snapshot snapshot(Ptr device) {
  static auto function = reinterpret_cast<U64 (*)(Ptr, Ptr)>(dlsym(RTLD_DEFAULT, "objc_msgSend"));
  if (!function || !device)
    std::abort();
  // Both selectors return NSUInteger/uint64_t on the 64-bit Mac runner.
  return {function(device, selector("currentAllocatedSize")),
          function(device, selector("recommendedMaxWorkingSetSize"))};
}

static Ptr physical_device(Ptr physical) {
  static auto function = original<void (*)(Ptr, Ptr *)>("vkGetMTLDeviceMVK");
  Ptr device = nullptr;
  function(physical, &device);
  if (!device)
    std::abort();
  return device;
}

static void remember(Ptr vulkan, Ptr metal) {
  std::lock_guard<std::mutex> guard(device_mutex);
  if (!devices.emplace(vulkan, metal).second)
    std::abort();
}

static Ptr forget(Ptr vulkan) {
  std::lock_guard<std::mutex> guard(device_mutex);
  Ptr result = devices.at(vulkan);
  devices.erase(vulkan);
  return result;
}

static std::array<U32, 2> configuration() {
  static auto function = original<Result (*)(Ptr, void *, size_t *)>("vkGetMoltenVKConfigurationMVK");
  size_t bytes = 0;
  if (function(nullptr, nullptr, &bytes) != 0 || bytes < 148 || bytes > 4096)
    std::abort();
  std::vector<unsigned char> data(bytes);
  if (function(nullptr, data.data(), &bytes) != 0)
    std::abort();
  // MVKConfiguration's append-only ABI in v1.3.0, arm64/x86_64: useMTLHeap at 124, argument buffers at 144.
  // This is the global configuration getter, not a query of per-device layer-setting overrides or residency sets.
  std::array<U32, 2> result{};
  std::memcpy(&result[0], data.data() + 124, sizeof(U32));
  std::memcpy(&result[1], data.data() + 144, sizeof(U32));
  return result;
}
}  // namespace metal_stats
