#include "dlpack_funcs.h"

#include <algorithm>
#include <vector>

// nb::cast<std::string>() (torch version check below) is compiled in this translation unit, so the std::string
// type caster must be visible here. Without it nanobind treats std::string as an unregistered bound type and
// nb::cast throws std::bad_cast at runtime.
#include <nanobind/stl/string.h>

#include "dlpack/dlpack.h"

#include "quadrants/program/ndarray.h"
#include "quadrants/program/program.h"
#if QD_WITH_CUDA
#include "quadrants/rhi/cuda/cuda_device.h"
#endif  // QD_WITH_CUDA
#if QD_WITH_AMDGPU
#include "quadrants/rhi/amdgpu/amdgpu_device.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#endif  // QD_WITH_AMDGPU
#if QD_WITH_METAL
#include "quadrants/rhi/metal/metal_device.h"
#endif  // QD_WITH_METAL
#include "quadrants/rhi/cpu/cpu_device.h"

namespace quadrants {
namespace lang {

namespace nb = nanobind;

void validate_arch(Arch arch) {
  if (!arch_is_cpu(arch) && !arch_is_cuda(arch) && !arch_is_metal(arch) && !arch_is_amdgpu(arch)) {
    QD_ERROR(
        "DLPack conversion is only supported on CPU, Metal, CUDA or AMDGPU "
        "archs");
  }
}

bool check_torch_version_lte(int major, int minor, int patch) {
  nb::module_ torch = nb::module_::import_("torch");
  std::string version = nb::cast<std::string>(torch.attr("__version__"));

  // Parse version string (e.g., "2.9.1" or "2.9.1+cu118")
  int vmajor = 0, vminor = 0, vpatch = 0;
  sscanf(version.c_str(), "%d.%d.%d", &vmajor, &vminor, &vpatch);

  if (vmajor < major)
    return true;
  if (vmajor > major)
    return false;
  if (vminor < minor)
    return true;
  if (vminor > minor)
    return false;
  return vpatch <= patch;
}

bool torch_supports_byte_offset() {
  static bool checked = false;
  static bool supports = false;
  if (!checked) {
    supports = !check_torch_version_lte(2, 9, 1);
    checked = true;
  }
  return supports;
}

std::tuple<void *, DLDeviceType> get_raw_ptr(Arch arch, Program *program, DeviceAllocation dev_alloc) {
  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;
  if (arch_is_cpu(arch)) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLCPU;
    cpu::CpuDevice::AllocInfo alloc_info = cpu_device->get_alloc_info(dev_alloc);
    raw_ptr = alloc_info.ptr;
  }
#if QD_WITH_CUDA
  else if (arch_is_cuda(arch)) {
    cuda::CudaDevice *cuda_device = static_cast<cuda::CudaDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLCUDA;
    cuda::CudaDevice::AllocInfo alloc_info = cuda_device->get_alloc_info(dev_alloc);
    raw_ptr = alloc_info.ptr;
  }
#endif  // QD_WITH_CUDA
#if QD_WITH_AMDGPU
  else if (arch_is_amdgpu(arch)) {
    amdgpu::AmdgpuDevice *amdgpu_device = static_cast<amdgpu::AmdgpuDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLROCM;  // AMDGPU uses the same device type as CUDA
    amdgpu::AmdgpuDevice::AllocInfo alloc_info = amdgpu_device->get_alloc_info(dev_alloc);
    raw_ptr = alloc_info.ptr;
  }
#endif  // QD_WITH_AMDGPU
#if QD_WITH_METAL
  else if (arch_is_metal(arch)) {
    metal::MetalDevice *metal_device = static_cast<metal::MetalDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLMetal;
    const metal::MetalMemory &memory = metal_device->get_memory(dev_alloc.alloc_id);

    RhiResult result = memory.mapped_ptr(&raw_ptr);
    if (result != RhiResult::success || raw_ptr == nullptr) {
      MTLBuffer_id mtl_buffer = memory.mtl_buffer();
      raw_ptr = mtl_buffer;
    }
  }
#endif  // QD_WITH_METAL

  if (raw_ptr == nullptr) {
    QD_ERROR("Unsupported device type for DLPack conversion");
  }
  return std::make_tuple(raw_ptr, device_type);
}

std::pair<uint8_t, uint8_t> get_type_info(Arch arch, DataType dt) {
  PrimitiveType *dt_as_primitive = dt->as<PrimitiveType>();
  if (dt_as_primitive == nullptr) {
    QD_ERROR("unsupported non-primitive data type for dlpack");
  }
  PrimitiveTypeID type_id = dt_as_primitive->type;
  uint8_t data_type_code = kDLInt;
  uint8_t element_bits = 0;
  switch (type_id) {
    case PrimitiveTypeID::i32: {
      data_type_code = static_cast<uint8_t>(kDLInt);
      element_bits = 32;
      break;
    }
    case PrimitiveTypeID::i64: {
      data_type_code = static_cast<uint8_t>(kDLInt);
      element_bits = 64;
      break;
    }
    case PrimitiveTypeID::f32: {
      data_type_code = static_cast<uint8_t>(kDLFloat);
      element_bits = 32;
      break;
    }
    case PrimitiveTypeID::f64: {
      data_type_code = static_cast<uint8_t>(kDLFloat);
      element_bits = 64;
      break;
    }
    case PrimitiveTypeID::u1: {
      data_type_code = static_cast<uint8_t>(kDLBool);
      element_bits = 8;
      break;
    }
    default: {
      QD_ERROR("unsupported ndarray data type for dlpack");
    }
  }
  return std::make_pair(data_type_code, element_bits);
}

// Walk the SNode chain root -> place and return the memory-layout permutation as a list of canonical-axis indices in
// the order they appear in physical memory (outermost first). For a 2-D field allocated with ``order='ji'`` this
// returns ``{1, 0}``; for ``order='ij'`` (or no ``order=``) it returns ``{0, 1}``.
//
// The returned vector is exactly the ``layout`` permutation accepted by ``ndarray_to_dlpack``, so the canonicalising
// shape/stride code path can be shared between the two backends.
std::vector<int> extract_memory_layout_order(SNode *snode) {
  std::vector<int> memory_layout_order;
  std::vector<const SNode *> path;
  const SNode *current = snode;
  while (current != nullptr) {
    path.push_back(current);
    current = current->parent;
  }
  std::reverse(path.begin(), path.end());  // Now path is root -> ... -> place

  for (const SNode *node : path) {
    if (node->type == SNodeType::dense) {
      for (int phys_axis = 0; phys_axis < quadrants_max_num_indices; phys_axis++) {
        if (node->extractors[phys_axis].active) {
          bool was_in_parent = false;
          if (node->parent != nullptr) {
            was_in_parent = node->parent->extractors[phys_axis].active;
          }
          if (!was_in_parent) {
            memory_layout_order.push_back(phys_axis);
          }
        }
      }
    }
  }
  return memory_layout_order;
}

int64_t *calc_strides(int64_t *shape, int full_ndim) {
  int64_t *strides = nullptr;
  if (full_ndim > 0) {
    strides = new int64_t[full_ndim];
    strides[full_ndim - 1] = 1;
    for (int i = full_ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  return strides;
}

nb::capsule field_to_dlpack(Program *program, SNode *snode, int element_ndim, int n, int m, bool versioned) {
  if (!snode->is_path_all_dense) {
    QD_ERROR("Only dense fields are supported for dlpack conversion");
  }

  Arch arch = program->compile_config().arch;
  validate_arch(arch);

#if QD_WITH_AMDGPU
  std::unique_ptr<AMDGPUContext::ContextGuard> amdgpu_guard;
  if (arch_is_amdgpu(arch)) {
    amdgpu_guard = std::make_unique<AMDGPUContext::ContextGuard>(&AMDGPUContext::get_instance());
  }
#endif

  int tree_id = snode->get_snode_tree_id();
  DevicePtr tree_device_ptr = program->get_snode_tree_device_ptr(tree_id);

  if (tree_device_ptr.device == nullptr || tree_device_ptr.alloc_id == 0) {
    QD_ERROR(
        "Field memory is not allocated. Please run 'ti.sync' before "
        "'to_dlpack'.")
  }

  int field_in_tree_offset = program->get_field_in_tree_offset(tree_id, snode);

  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;
  std::tie(raw_ptr, device_type) = get_raw_ptr(arch, program, tree_device_ptr);

  int byte_offset = 0;
  if (field_in_tree_offset >= 0) {
    if (torch_supports_byte_offset()) {
      byte_offset = field_in_tree_offset;
    } else {
      if (arch_is_metal(arch)) {
        QD_ERROR(
            "DLPack conversion with fields is not supported on Metal "
            "with PyTorch <= 2.9.1.");
      }
      raw_ptr = reinterpret_cast<void *>((uint64_t)raw_ptr + field_in_tree_offset);
    }
  }

  DataType dt = snode->dt;

  uint8_t element_bits = 32;
  uint8_t data_type_code = kDLInt;
  std::tie(data_type_code, element_bits) = get_type_info(arch, dt);

  int ndim = snode->num_active_indices;

  // Derive the field's physical-memory axis order from the SNode chain. For a 2-D field with ``order='ji'`` this
  // yields ``{1, 0}``; for the default (no ``order=``) it yields ``{0, 1, ..., ndim-1}``.
  //
  // The element axes (``n``, ``m`` for VectorField/MatrixField) always sit innermost and are never permuted, so we
  // extend ``layout`` with identity entries to cover them.
  std::vector<int> mem_layout = extract_memory_layout_order(snode);
  if (static_cast<int>(mem_layout.size()) != ndim) {
    QD_ERROR("field_to_dlpack: SNode chain produced %d memory axes for a %d-D field",
             static_cast<int>(mem_layout.size()), ndim);
  }
  // mem_layout must be a permutation of {0, 1, ..., ndim-1}. Fields built with non-contiguous axis identifiers (e.g.
  // qd.i + qd.l, skipping qd.j and qd.k) are rejected — the canonical view is undefined.
  {
    std::vector<int> sorted_layout = mem_layout;
    std::sort(sorted_layout.begin(), sorted_layout.end());
    for (int i = 0; i < ndim; i++) {
      if (sorted_layout[i] != i) {
        QD_ERROR("field_to_dlpack: SNode axes must be a contiguous permutation of {0, ..., ndim-1}");
      }
    }
  }

  int full_ndim = ndim + element_ndim;
  int64_t *shape = nullptr;
  int64_t *strides = nullptr;
  if (full_ndim > 0) {
    // Build the *physical* shape and strides first (memory order).
    int64_t *phys_shape = new int64_t[full_ndim];
    for (int i = 0; i < ndim; i++) {
      phys_shape[i] = snode->shape_along_axis(mem_layout[i]);
    }
    if (element_ndim >= 1) {
      phys_shape[ndim] = n;
    }
    if (element_ndim == 2) {
      phys_shape[ndim + 1] = m;
    }
    int64_t *phys_strides = calc_strides(phys_shape, full_ndim);

    // Now apply the inverse permutation to expose a *canonical* view to the DLPack consumer: ``shape[j]`` is the
    // canonical-axis-j extent, ``strides[j]`` is its physical-memory stride. Element axes (last ``element_ndim``
    // entries) are identity, never permuted.
    bool is_identity = true;
    for (int i = 0; i < ndim; i++) {
      if (mem_layout[i] != i) {
        is_identity = false;
        break;
      }
    }
    if (is_identity) {
      shape = phys_shape;
      strides = phys_strides;
    } else {
      shape = new int64_t[full_ndim];
      strides = new int64_t[full_ndim];
      // invperm[canonical_axis] = physical_axis
      std::vector<int> invperm(ndim, 0);
      for (int i = 0; i < ndim; i++) {
        invperm[mem_layout[i]] = i;
      }
      for (int j = 0; j < ndim; j++) {
        shape[j] = phys_shape[invperm[j]];
        strides[j] = phys_strides[invperm[j]];
      }
      for (int k = 0; k < element_ndim; k++) {
        shape[ndim + k] = phys_shape[ndim + k];
        strides[ndim + k] = phys_strides[ndim + k];
      }
      delete[] phys_shape;
      delete[] phys_strides;
    }
  }

  DLTensor dl_tensor;
  dl_tensor.data = raw_ptr;
  dl_tensor.device.device_type = device_type;
  dl_tensor.device.device_id = 0;
  dl_tensor.ndim = full_ndim;
  dl_tensor.dtype = DLDataType{data_type_code, element_bits, 1};
  dl_tensor.shape = shape;
  dl_tensor.strides = strides;
  dl_tensor.byte_offset = byte_offset;

  if (versioned) {
    auto *vt = new DLManagedTensorVersioned();
    vt->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    vt->manager_ctx = nullptr;
    vt->flags = 0;
    vt->dl_tensor = dl_tensor;
    vt->deleter = [](DLManagedTensorVersioned *self) {
      if (self->dl_tensor.shape != nullptr) {
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
      }
      delete self;
    };
    auto capsule_deleter = [](PyObject *capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        auto *vt = reinterpret_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(capsule, "dltensor_versioned"));
        if (vt && vt->deleter)
          vt->deleter(vt);
      }
    };
    return nb::steal<nb::capsule>(PyCapsule_New(vt, "dltensor_versioned", +capsule_deleter));
  }

  auto *mt = new DLManagedTensor();
  mt->dl_tensor = dl_tensor;
  mt->manager_ctx = nullptr;
  mt->deleter = [](DLManagedTensor *self) {
    if (self->dl_tensor.shape != nullptr) {
      delete[] self->dl_tensor.shape;
      delete[] self->dl_tensor.strides;
    }
    delete self;
  };
  auto capsule_deleter = [](PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
      auto *mt = reinterpret_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
      if (mt && mt->deleter)
        mt->deleter(mt);
    }
  };
  return nb::steal<nb::capsule>(PyCapsule_New(mt, "dltensor", +capsule_deleter));
}

nb::capsule ndarray_to_dlpack(Program *program,
                                    nb::object owner,
                                    Ndarray *ndarray,
                                    const std::vector<int> &layout,
                                    bool versioned) {
  Arch arch = program->compile_config().arch;
  validate_arch(arch);

#if QD_WITH_AMDGPU
  std::unique_ptr<AMDGPUContext::ContextGuard> amdgpu_guard;
  if (arch_is_amdgpu(arch)) {
    amdgpu_guard = std::make_unique<AMDGPUContext::ContextGuard>(&AMDGPUContext::get_instance());
  }
#endif

  auto *owner_holder = new nb::object(owner);

  DeviceAllocation devalloc = ndarray->get_device_allocation();

  DLDeviceType device_type = DLDeviceType::kDLCPU;
  void *raw_ptr = nullptr;
  std::tie(raw_ptr, device_type) = get_raw_ptr(arch, program, devalloc);

  // ``ndarray_shape`` is the *physical* (storage) shape of the buffer. When ``layout`` is non-empty it lists the
  // canonical-axis index at each successive physical-memory axis (outermost first). We expose a *canonical* view to
  // DLPack consumers by permuting both the shape and the strides via the inverse permutation, leaving the raw pointer
  // and byte offset untouched (no data movement).
  std::vector<int> ndarray_shape = ndarray->total_shape();
  int ndim = ndarray_shape.size();

  if (!layout.empty() && static_cast<int>(layout.size()) != ndim) {
    QD_ERROR("ndarray_to_dlpack: layout has wrong size for this ndarray");
  }

  int64_t *shape = nullptr;
  int64_t *strides = nullptr;
  if (ndim > 0) {
    int64_t *phys_shape = new int64_t[ndim];
    std::copy(ndarray_shape.begin(), ndarray_shape.end(), phys_shape);
    int64_t *phys_strides = calc_strides(phys_shape, ndim);

    if (layout.empty()) {
      shape = phys_shape;
      strides = phys_strides;
    } else {
      // Build the inverse permutation: invperm[layout[i]] = i.
      std::vector<int> invperm(ndim, 0);
      for (int i = 0; i < ndim; i++) {
        if (layout[i] < 0 || layout[i] >= ndim) {
          delete[] phys_shape;
          delete[] phys_strides;
          QD_ERROR("ndarray_to_dlpack: layout entry out of range");
        }
        invperm[layout[i]] = i;
      }
      shape = new int64_t[ndim];
      strides = new int64_t[ndim];
      for (int j = 0; j < ndim; j++) {
        shape[j] = phys_shape[invperm[j]];
        strides[j] = phys_strides[invperm[j]];
      }
      delete[] phys_shape;
      delete[] phys_strides;
    }
  }

  DataType ndarray_data_type = ndarray->get_element_data_type();
  uint8_t data_type_code = kDLInt;
  uint8_t element_bits = 0;
  std::tie(data_type_code, element_bits) = get_type_info(arch, ndarray_data_type);

  DLTensor dl_tensor;
  dl_tensor.data = raw_ptr;
  dl_tensor.device.device_type = device_type;
  dl_tensor.device.device_id = 0;
  dl_tensor.ndim = ndim;
  dl_tensor.dtype = DLDataType{data_type_code, element_bits, 1};
  dl_tensor.shape = shape;
  dl_tensor.strides = strides;
  dl_tensor.byte_offset = 0;

  if (versioned) {
    auto *vt = new DLManagedTensorVersioned();
    vt->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    vt->manager_ctx = owner_holder;
    vt->flags = 0;
    vt->dl_tensor = dl_tensor;
    vt->deleter = [](DLManagedTensorVersioned *self) {
      auto *owner = reinterpret_cast<nb::object *>(self->manager_ctx);
      nb::gil_scoped_acquire gil;
      delete owner;  // DECREFs the Python object
      if (self->dl_tensor.shape != nullptr) {
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
      }
      delete self;
    };
    auto capsule_deleter = [](PyObject *capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        auto *vt = reinterpret_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(capsule, "dltensor_versioned"));
        if (vt && vt->deleter)
          vt->deleter(vt);
      }
    };
    return nb::steal<nb::capsule>(PyCapsule_New(vt, "dltensor_versioned", +capsule_deleter));
  }

  auto *mt = new DLManagedTensor();
  mt->dl_tensor = dl_tensor;
  mt->manager_ctx = owner_holder;
  mt->deleter = [](DLManagedTensor *self) {
    auto *owner = reinterpret_cast<nb::object *>(self->manager_ctx);
    nb::gil_scoped_acquire gil;
    delete owner;  // DECREFs the Python object
    if (self->dl_tensor.shape != nullptr) {
      delete[] self->dl_tensor.shape;
      delete[] self->dl_tensor.strides;
    }
    delete self;
  };
  auto capsule_deleter = [](PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
      auto *mt = reinterpret_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
      if (mt && mt->deleter)
        mt->deleter(mt);
    }
  };
  return nb::steal<nb::capsule>(PyCapsule_New(mt, "dltensor", +capsule_deleter));
}
}  // namespace lang
}  // namespace quadrants
