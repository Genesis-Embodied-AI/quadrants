// Diagnostic forwarding library for the archived MoltenVK binary. Numeric Vulkan ABI layouts below follow
// Vulkan-Headers v1.3.290. Calls and results are forwarded unchanged; JSONL records describe actual allocations.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <unistd.h>

using U32 = uint32_t;
using U64 = uint64_t;
using Ptr = void *;
using Result = int32_t;
struct Header {
  U32 type;
  const void *next;
};
struct BufferInfo {
  U32 type;
  const void *next;
  U32 flags;
  U64 size;
  U32 usage;
  U32 sharing;
  U32 count;
  const U32 *families;
};
struct AllocateInfo {
  U32 type;
  const void *next;
  U64 size;
  U32 memory_type;
};
struct AllocateFlags {
  U32 type;
  const void *next;
  U32 flags;
  U32 device_mask;
};
struct Requirements {
  U64 size;
  U64 alignment;
  U32 bits;
};
struct BufferReference {
  U32 type;
  const void *next;
  U64 buffer;
};
struct Requirements2 {
  U32 type;
  void *next;
  Requirements requirements;
};
struct BindInfo {
  U32 type;
  const void *next;
  U64 buffer;
  U64 memory;
  U64 offset;
};

static void *library() {
  static void *handle = [] {
    const char *path = std::getenv("QD_TRACE_MOLTENVK_REAL");
    if (!path)
      std::abort();
    void *result = dlopen(path, RTLD_LOCAL | RTLD_NOW);
    if (!result) {
      std::fprintf(stderr, "trace dlopen: %s\n", dlerror());
      std::abort();
    }
    return result;
  }();
  return handle;
}

template <class T>
static T original(const char *name) {
  auto result = reinterpret_cast<T>(dlsym(library(), name));
  if (!result) {
    std::fprintf(stderr, "trace missing %s\n", name);
    std::abort();
  }
  return result;
}

static void record(const char *format, ...) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  static FILE *stream = [] {
    const char *folder = std::getenv("QD_TRACE_VULKAN_DIR");
    if (!folder)
      std::abort();
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/api-%d.jsonl", folder, getpid());
    FILE *file = std::fopen(path, "a");
    if (!file)
      std::abort();
    std::setvbuf(file, nullptr, _IOLBF, 0);
    return file;
  }();
  va_list args;
  va_start(args, format);
  std::vfprintf(stream, format, args);
  va_end(args);
  std::fputc('\n', stream);
}

static Result create_device(Ptr physical, const void *info, const void *callbacks, Ptr *device) {
  static auto fn = original<Result (*)(Ptr, const void *, const void *, Ptr *)>("vkCreateDevice");
  Result result = fn(physical, info, callbacks, device);
  record("{\"call\":\"create_device\",\"result\":%d,\"device\":%llu}", result,
         result == 0 ? (unsigned long long)(uintptr_t)*device : 0ULL);
  return result;
}
static void destroy_device(Ptr device, const void *callbacks) {
  static auto fn = original<void (*)(Ptr, const void *)>("vkDestroyDevice");
  record("{\"call\":\"destroy_device\",\"device\":%llu}", (unsigned long long)(uintptr_t)device);
  fn(device, callbacks);
}
static Result create_buffer(Ptr device, const BufferInfo *info, const void *callbacks, U64 *buffer) {
  static auto fn = original<Result (*)(Ptr, const BufferInfo *, const void *, U64 *)>("vkCreateBuffer");
  Result result = fn(device, info, callbacks, buffer);
  record("{\"call\":\"create_buffer\",\"result\":%d,\"buffer\":%llu,\"bytes\":%llu,\"usage\":%u,\"flags\":%u}", result,
         result == 0 ? (unsigned long long)*buffer : 0ULL, (unsigned long long)info->size, info->usage, info->flags);
  return result;
}
static void destroy_buffer(Ptr device, U64 buffer, const void *callbacks) {
  static auto fn = original<void (*)(Ptr, U64, const void *)>("vkDestroyBuffer");
  record("{\"call\":\"destroy_buffer\",\"buffer\":%llu}", (unsigned long long)buffer);
  fn(device, buffer, callbacks);
}
static Result allocate_memory(Ptr device, const AllocateInfo *info, const void *callbacks, U64 *memory) {
  static auto fn = original<Result (*)(Ptr, const AllocateInfo *, const void *, U64 *)>("vkAllocateMemory");
  U32 flags = 0;
  for (auto h = reinterpret_cast<const Header *>(info->next); h; h = reinterpret_cast<const Header *>(h->next))
    if (h->type == 1000060000)
      flags = reinterpret_cast<const AllocateFlags *>(h)->flags;
  Result result = fn(device, info, callbacks, memory);
  record("{\"call\":\"allocate_memory\",\"result\":%d,\"memory\":%llu,\"bytes\":%llu,\"type\":%u,\"flags\":%u}", result,
         result == 0 ? (unsigned long long)*memory : 0ULL, (unsigned long long)info->size, info->memory_type, flags);
  return result;
}
static void free_memory(Ptr device, U64 memory, const void *callbacks) {
  static auto fn = original<void (*)(Ptr, U64, const void *)>("vkFreeMemory");
  record("{\"call\":\"free_memory\",\"memory\":%llu}", (unsigned long long)memory);
  fn(device, memory, callbacks);
}
static void report_requirements(U64 buffer, const Requirements &req) {
  record("{\"call\":\"requirements\",\"buffer\":%llu,\"bytes\":%llu,\"alignment\":%llu,\"type_bits\":%u}",
         (unsigned long long)buffer, (unsigned long long)req.size, (unsigned long long)req.alignment, req.bits);
}
static void requirements(Ptr device, U64 buffer, Requirements *req) {
  static auto fn = original<void (*)(Ptr, U64, Requirements *)>("vkGetBufferMemoryRequirements");
  fn(device, buffer, req);
  report_requirements(buffer, *req);
}
static void requirements2(Ptr device, const BufferReference *info, Requirements2 *req) {
  static auto fn = original<void (*)(Ptr, const BufferReference *, Requirements2 *)>("vkGetBufferMemoryRequirements2");
  fn(device, info, req);
  report_requirements(info->buffer, req->requirements);
}
static Result bind_memory(Ptr device, U64 buffer, U64 memory, U64 offset) {
  static auto fn = original<Result (*)(Ptr, U64, U64, U64)>("vkBindBufferMemory");
  Result result = fn(device, buffer, memory, offset);
  record("{\"call\":\"bind\",\"result\":%d,\"buffer\":%llu,\"memory\":%llu,\"offset\":%llu}", result,
         (unsigned long long)buffer, (unsigned long long)memory, (unsigned long long)offset);
  return result;
}
static Result bind_memory2(Ptr device, U32 count, const BindInfo *infos) {
  static auto fn = original<Result (*)(Ptr, U32, const BindInfo *)>("vkBindBufferMemory2");
  Result result = fn(device, count, infos);
  for (U32 i = 0; i < count; ++i)
    record("{\"call\":\"bind\",\"result\":%d,\"buffer\":%llu,\"memory\":%llu,\"offset\":%llu}", result,
           (unsigned long long)infos[i].buffer, (unsigned long long)infos[i].memory,
           (unsigned long long)infos[i].offset);
  return result;
}
static U64 address(Ptr device, const BufferReference *info) {
  static auto fn = original<U64 (*)(Ptr, const BufferReference *)>("vkGetBufferDeviceAddress");
  U64 result = fn(device, info);
  record("{\"call\":\"address\",\"buffer\":%llu,\"nonzero\":%s}", (unsigned long long)info->buffer,
         result ? "true" : "false");
  return result;
}
extern "C" void *vkGetInstanceProcAddr(Ptr instance, const char *name);
extern "C" void *vkGetDeviceProcAddr(Ptr device, const char *name);
static void *intercept(const char *name, void *real) {
  if (!real)
    return nullptr;
#define HOOK(vk_name, fn)              \
  if (std::strcmp(name, vk_name) == 0) \
  return reinterpret_cast<void *>(&fn)
  HOOK("vkGetInstanceProcAddr", vkGetInstanceProcAddr);
  HOOK("vkGetDeviceProcAddr", vkGetDeviceProcAddr);
  HOOK("vkCreateDevice", create_device);
  HOOK("vkDestroyDevice", destroy_device);
  HOOK("vkCreateBuffer", create_buffer);
  HOOK("vkDestroyBuffer", destroy_buffer);
  HOOK("vkAllocateMemory", allocate_memory);
  HOOK("vkFreeMemory", free_memory);
  HOOK("vkGetBufferMemoryRequirements", requirements);
  HOOK("vkGetBufferMemoryRequirements2", requirements2);
  HOOK("vkGetBufferMemoryRequirements2KHR", requirements2);
  HOOK("vkBindBufferMemory", bind_memory);
  HOOK("vkBindBufferMemory2", bind_memory2);
  HOOK("vkBindBufferMemory2KHR", bind_memory2);
  HOOK("vkGetBufferDeviceAddress", address);
  HOOK("vkGetBufferDeviceAddressKHR", address);
#undef HOOK
  return real;
}
extern "C" void *vkGetInstanceProcAddr(Ptr instance, const char *name) {
  static auto fn = original<void *(*)(Ptr, const char *)>("vkGetInstanceProcAddr");
  return intercept(name, fn(instance, name));
}
extern "C" void *vkGetDeviceProcAddr(Ptr device, const char *name) {
  static auto fn = original<void *(*)(Ptr, const char *)>("vkGetDeviceProcAddr");
  return intercept(name, fn(device, name));
}
