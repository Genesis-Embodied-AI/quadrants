#include "objc_api.h"

#ifdef QD_PLATFORM_OSX

namespace quadrants {
namespace mac {

nsobj_unique_ptr<QD_NSString> wrap_string_as_ns_string(const std::string &str) {
  constexpr int kNSUTF8StringEncoding = 4;
  id ns_string = clscall("NSString", "alloc");
  auto *ptr = cast_call<QD_NSString *>(
      ns_string,
      "initWithBytesNoCopy:length:encoding:freeWhenDone:", str.data(),
      str.size(), kNSUTF8StringEncoding, false);
  return wrap_as_nsobj_unique_ptr(ptr);
}

std::string to_string(QD_NSString *ns) {
  return cast_call<const char *>(ns, "UTF8String");
}

int ns_array_count(QD_NSArray *na) {
  return cast_call<int>(na, "count");
}

QD_NSAutoreleasePool *create_autorelease_pool() {
  return cast_call<QD_NSAutoreleasePool *>(
      clscall("NSAutoreleasePool", "alloc"), "init");
}

void drain_autorelease_pool(QD_NSAutoreleasePool *pool) {
  // "drain" is same as "release", so we don't need to release |pool| itself.
  // https://developer.apple.com/documentation/foundation/nsautoreleasepool
  call(pool, "drain");
}

ScopedAutoreleasePool::ScopedAutoreleasePool() {
  pool_ = create_autorelease_pool();
}

ScopedAutoreleasePool::~ScopedAutoreleasePool() {
  drain_autorelease_pool(pool_);
}

}  // namespace mac
}  // namespace quadrants

#endif  // QD_PLATFORM_OSX
