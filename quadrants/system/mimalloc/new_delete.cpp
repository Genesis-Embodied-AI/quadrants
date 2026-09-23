/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

// Replacement global operator new / delete, backed by mimalloc. Only compiled when QD_USE_MIMALLOC is enabled (see
// cmake/QuadrantsMimalloc.cmake). These definitions take precedence over the CRT's for everything linked into the
// same DLL / EXE, but not for other modules in the process.
//
// A pointer freed here may still come from the CRT heap, e.g. if another module allocated it and handed over
// ownership. Such pointers are returned to the CRT instead of being passed to mimalloc.

#include <cstdlib>
#include <new>

#include <malloc.h>
#include <mimalloc.h>

namespace {

inline void free_any(void *p) noexcept {
  if (p == nullptr) {
    return;
  }
  if (mi_is_in_heap_region(p)) {
    mi_free(p);
  } else {
    std::free(p);
  }
}

inline void free_any_sized(void *p, std::size_t n) noexcept {
  if (p == nullptr) {
    return;
  }
  if (mi_is_in_heap_region(p)) {
    mi_free_size(p, n);
  } else {
    std::free(p);
  }
}

// The CRT allocates over-aligned objects with _aligned_malloc, which must be released with _aligned_free.
inline void free_any_aligned(void *p, std::align_val_t al) noexcept {
  if (p == nullptr) {
    return;
  }
  if (mi_is_in_heap_region(p)) {
    mi_free_aligned(p, static_cast<std::size_t>(al));
  } else {
    _aligned_free(p);
  }
}

inline void free_any_sized_aligned(void *p, std::size_t n, std::align_val_t al) noexcept {
  if (p == nullptr) {
    return;
  }
  if (mi_is_in_heap_region(p)) {
    mi_free_size_aligned(p, n, static_cast<std::size_t>(al));
  } else {
    _aligned_free(p);
  }
}

}  // namespace

void *operator new(std::size_t n) {
  return mi_new(n);
}
void *operator new[](std::size_t n) {
  return mi_new(n);
}
void *operator new(std::size_t n, const std::nothrow_t &) noexcept {
  return mi_new_nothrow(n);
}
void *operator new[](std::size_t n, const std::nothrow_t &) noexcept {
  return mi_new_nothrow(n);
}
void *operator new(std::size_t n, std::align_val_t al) {
  return mi_new_aligned(n, static_cast<std::size_t>(al));
}
void *operator new[](std::size_t n, std::align_val_t al) {
  return mi_new_aligned(n, static_cast<std::size_t>(al));
}
void *operator new(std::size_t n, std::align_val_t al, const std::nothrow_t &) noexcept {
  return mi_new_aligned_nothrow(n, static_cast<std::size_t>(al));
}
void *operator new[](std::size_t n, std::align_val_t al, const std::nothrow_t &) noexcept {
  return mi_new_aligned_nothrow(n, static_cast<std::size_t>(al));
}

void operator delete(void *p) noexcept {
  free_any(p);
}
void operator delete[](void *p) noexcept {
  free_any(p);
}
void operator delete(void *p, const std::nothrow_t &) noexcept {
  free_any(p);
}
void operator delete[](void *p, const std::nothrow_t &) noexcept {
  free_any(p);
}
void operator delete(void *p, std::size_t n) noexcept {
  free_any_sized(p, n);
}
void operator delete[](void *p, std::size_t n) noexcept {
  free_any_sized(p, n);
}
void operator delete(void *p, std::align_val_t al) noexcept {
  free_any_aligned(p, al);
}
void operator delete[](void *p, std::align_val_t al) noexcept {
  free_any_aligned(p, al);
}
void operator delete(void *p, std::align_val_t al, const std::nothrow_t &) noexcept {
  free_any_aligned(p, al);
}
void operator delete[](void *p, std::align_val_t al, const std::nothrow_t &) noexcept {
  free_any_aligned(p, al);
}
void operator delete(void *p, std::size_t n, std::align_val_t al) noexcept {
  free_any_sized_aligned(p, n, al);
}
void operator delete[](void *p, std::size_t n, std::align_val_t al) noexcept {
  free_any_sized_aligned(p, n, al);
}
