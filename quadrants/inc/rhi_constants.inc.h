// (penguinliong) Device capability is a shared intelligence between the runtime
// environment and the code generator. It's only about the program executed
// on-device rather than any platform specific capability provided by some
// graphics APIs like Vulkan or CUDA. For example, DirectX shader model, CUDA
// compute capability and Vulkan physical device features can be listed here as
// device capabilities, yet things like Vulkan API version
// should not.
#ifdef PER_DEVICE_CAPABILITY
// SPIR-V Caps
PER_DEVICE_CAPABILITY(reserved)
PER_DEVICE_CAPABILITY(spirv_version)
PER_DEVICE_CAPABILITY(spirv_has_int8)
PER_DEVICE_CAPABILITY(spirv_has_int16)
PER_DEVICE_CAPABILITY(spirv_has_int64)
PER_DEVICE_CAPABILITY(spirv_has_float16)
PER_DEVICE_CAPABILITY(spirv_has_float64)
// Vulkan `VkPhysicalDevice8BitStorageFeatures::storageBuffer8BitAccess` /
// `VkPhysicalDevice16BitStorageFeatures::storageBuffer16BitAccess` queried flags, enabling
// `CapabilityStorageBuffer{8,16}BitAccess` emission in the SPIR-V header. Needed independently of
// `spirv_has_int{8,16}` / `spirv_has_float16`: `Int8` / `Int16` / `Float16` capabilities gate
// whether the type can be declared, whereas these gate whether it can be loaded / stored through
// a descriptor-bound `StorageBuffer` pointer. Emitting the storage caps without the underlying
// device feature makes the driver reject the shader at pipeline creation on strict
// implementations; skipping them when the device does support them is silent-corruption-UB on
// drivers that enforce them.
PER_DEVICE_CAPABILITY(spirv_has_storage_buffer_8bit_access)
PER_DEVICE_CAPABILITY(spirv_has_storage_buffer_16bit_access)
PER_DEVICE_CAPABILITY(spirv_has_atomic_int64)
PER_DEVICE_CAPABILITY(spirv_has_atomic_float16)  // load, store, exchange
PER_DEVICE_CAPABILITY(spirv_has_atomic_float16_add)
PER_DEVICE_CAPABILITY(spirv_has_atomic_float16_minmax)
PER_DEVICE_CAPABILITY(spirv_has_atomic_float)  // load, store, exchange
PER_DEVICE_CAPABILITY(spirv_has_atomic_float_add)
PER_DEVICE_CAPABILITY(spirv_has_atomic_float_minmax)
PER_DEVICE_CAPABILITY(spirv_has_atomic_float64)  // load, store, exchange
PER_DEVICE_CAPABILITY(spirv_has_atomic_float64_add)
PER_DEVICE_CAPABILITY(spirv_has_atomic_float64_minmax)
PER_DEVICE_CAPABILITY(spirv_has_shared_atomic_float_add)
PER_DEVICE_CAPABILITY(spirv_has_shared_atomic_float64_add)
PER_DEVICE_CAPABILITY(spirv_has_shared_atomic_float16_add)
PER_DEVICE_CAPABILITY(spirv_has_variable_ptr)
PER_DEVICE_CAPABILITY(spirv_has_physical_storage_buffer)
PER_DEVICE_CAPABILITY(spirv_has_subgroup_basic)
PER_DEVICE_CAPABILITY(spirv_has_subgroup_vote)
PER_DEVICE_CAPABILITY(spirv_has_subgroup_arithmetic)
PER_DEVICE_CAPABILITY(spirv_has_subgroup_ballot)
// Device-probed subgroup size used by SPIR-V (Vulkan / Metal) backends. Stored as a positive integer
// (e.g. 32 on every Apple GPU and every desktop NVIDIA/AMD Vulkan driver we've tested). Always uniform
// at queue submit time on every implementation we target - Vulkan returns
// `VkPhysicalDeviceSubgroupProperties::subgroupSize` once at device creation; Metal's SIMD width is
// fixed per-device. Mixed in via `Program::subgroup_size()` for `qd.simt.subgroup.group_size()` and via
// the per-kernel `DeviceCapabilityConfig` cache key for fe-ll cache correctness. Unset / zero on
// LLVM (CUDA / AMDGPU / x64) backends - those return their respective constants from
// `Program::subgroup_size()` directly without consulting the cap.
PER_DEVICE_CAPABILITY(spirv_subgroup_size)
PER_DEVICE_CAPABILITY(spirv_has_non_semantic_info)
PER_DEVICE_CAPABILITY(spirv_has_no_integer_wrap_decoration)
PER_DEVICE_CAPABILITY(spirv_has_shader_clock)
#endif

#ifdef PER_BUFFER_FORMAT
PER_BUFFER_FORMAT(unknown)
PER_BUFFER_FORMAT(r8)
PER_BUFFER_FORMAT(rg8)
PER_BUFFER_FORMAT(rgba8)
PER_BUFFER_FORMAT(rgba8srgb)
PER_BUFFER_FORMAT(bgra8)
PER_BUFFER_FORMAT(bgra8srgb)
PER_BUFFER_FORMAT(r8u)
PER_BUFFER_FORMAT(rg8u)
PER_BUFFER_FORMAT(rgba8u)
PER_BUFFER_FORMAT(r8i)
PER_BUFFER_FORMAT(rg8i)
PER_BUFFER_FORMAT(rgba8i)
PER_BUFFER_FORMAT(r16)
PER_BUFFER_FORMAT(rg16)
PER_BUFFER_FORMAT(rgb16)
PER_BUFFER_FORMAT(rgba16)
PER_BUFFER_FORMAT(r16u)
PER_BUFFER_FORMAT(rg16u)
PER_BUFFER_FORMAT(rgb16u)
PER_BUFFER_FORMAT(rgba16u)
PER_BUFFER_FORMAT(r16i)
PER_BUFFER_FORMAT(rg16i)
PER_BUFFER_FORMAT(rgb16i)
PER_BUFFER_FORMAT(rgba16i)
PER_BUFFER_FORMAT(r16f)
PER_BUFFER_FORMAT(rg16f)
PER_BUFFER_FORMAT(rgb16f)
PER_BUFFER_FORMAT(rgba16f)
PER_BUFFER_FORMAT(r32u)
PER_BUFFER_FORMAT(rg32u)
PER_BUFFER_FORMAT(rgb32u)
PER_BUFFER_FORMAT(rgba32u)
PER_BUFFER_FORMAT(r32i)
PER_BUFFER_FORMAT(rg32i)
PER_BUFFER_FORMAT(rgb32i)
PER_BUFFER_FORMAT(rgba32i)
PER_BUFFER_FORMAT(r32f)
PER_BUFFER_FORMAT(rg32f)
PER_BUFFER_FORMAT(rgb32f)
PER_BUFFER_FORMAT(rgba32f)
PER_BUFFER_FORMAT(depth16)
PER_BUFFER_FORMAT(depth24stencil8)
PER_BUFFER_FORMAT(depth32f)
#endif

#ifdef PER_IMAGE_DIMENSION
PER_IMAGE_DIMENSION(d1D)
PER_IMAGE_DIMENSION(d2D)
PER_IMAGE_DIMENSION(d3D)
#endif

#ifdef PER_IMAGE_LAYOUT
PER_IMAGE_LAYOUT(undefined)
PER_IMAGE_LAYOUT(shader_read)
PER_IMAGE_LAYOUT(shader_write)
PER_IMAGE_LAYOUT(shader_read_write)
PER_IMAGE_LAYOUT(color_attachment)
PER_IMAGE_LAYOUT(color_attachment_read)
PER_IMAGE_LAYOUT(depth_attachment)
PER_IMAGE_LAYOUT(depth_attachment_read)
PER_IMAGE_LAYOUT(transfer_dst)
PER_IMAGE_LAYOUT(transfer_src)
PER_IMAGE_LAYOUT(present_src)
#endif
