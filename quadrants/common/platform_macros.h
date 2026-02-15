#pragma once

// TO KEEP THE INCLUDE DEPENDENCY CLEAN, PLEASE DO NOT INCLUDE ANY OTHER
// QUADRANTS HEADERS INTO THIS ONE.
//
// TODO(#2196): Once we can slim down "quadrants/common/core.h", consider moving
// the contents back to core.h and delete this file.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

// https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#ifdef __GNUC__
#define QD_DLL_EXPORT __attribute__((dllexport))
#else
#define QD_DLL_EXPORT __declspec(dllexport)
#endif  //  __GNUC__
#else
#define QD_DLL_EXPORT __attribute__((visibility("default")))
#endif  // defined _WIN32 || defined _WIN64 || defined __CYGWIN__

// Windows
#if defined(_WIN64)
#define QD_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#define QD_PLATFORM_LINUX
#endif

// OSX
#if defined(__APPLE__)
#define QD_PLATFORM_OSX
#endif

#if (defined(QD_PLATFORM_LINUX) || defined(QD_PLATFORM_OSX) || \
     defined(__unix__))
#define QD_PLATFORM_UNIX
#endif
