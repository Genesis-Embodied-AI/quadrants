#pragma once
#include <quadrants/program/callable.h>
#include "quadrants/program/ndarray.h"
#include "quadrants/program/matrix.h"

namespace quadrants::lang {

struct RuntimeContext;

// One entry per `graph_do_while` loop in a `graph=True` kernel, indexed by level id (the order the AST transformer
// assigns: outer levels before the inner levels they contain). For nested loops the runtime rebuilds the loop tree from
// these entries plus the per-task `graph_do_while_level_id` tags baked into each OffloadedTask. A non-nested (depth-1)
// kernel has exactly one entry with parent -1.
struct GraphDoWhileLevel {
  // Kernel parameter index (C++ arg id, post-template) of this level's condition ndarray.
  int cond_arg_id{-1};
  // Enclosing level id, or -1 if this level is at the kernel top level.
  int parent_id{-1};
  // Device pointer of the condition flag, resolved by the backend each launch from `cond_arg_id`.
  void *flag_dev_ptr{nullptr};
};

struct ArgArrayPtrKey {
  int32_t arg_id;
  int32_t ptr_type;
};

inline bool operator==(ArgArrayPtrKey lhs, ArgArrayPtrKey rhs) noexcept {
  return lhs.arg_id == rhs.arg_id && lhs.ptr_type == rhs.ptr_type;
}

struct ArgArrayPtrKeyHasher {
  size_t operator()(ArgArrayPtrKey k) const noexcept {
    return (static_cast<size_t>(static_cast<uint32_t>(k.arg_id)) << 32) | static_cast<uint32_t>(k.ptr_type);
  }
};

class LaunchContextBuilder {
 public:
  enum class DevAllocType : int8_t {
    kNone = 0,
    kNdarray = 1,
    // kArgPack = 4,
  };

  explicit LaunchContextBuilder(CallableBase *kernel);

  LaunchContextBuilder(LaunchContextBuilder &&) = default;
  LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
  LaunchContextBuilder(const LaunchContextBuilder &) = delete;
  LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

  // Copy all the arguments already added to an existing launcher context.
  // The input context must be associated with the exactly same kernel, and
  // the current context must be fresh new, without any variable already added.
  // This method is useful to speed up calling repeatedly a given kernel with
  // the exact same input arguments.
  void copy(const LaunchContextBuilder &other);

  void set_arg_float(int arg_id, float64 d);
  // Bulk processing of multiple scalar float arguments at the same time.
  // This is mainly useful to mitigate pybind11 function call overhead.
  // In this context, 'args_id' is a vector gathering the position of each
  // of these scalar arguments in the corresponding kernel. As a result, the
  // length 'args_id' and 'vec' must be equal.
  void set_args_float(const std::vector<int> &args_id, const std::vector<float64> &vec);

  // Created signed and unsigned version for argument range check of pybind
  void set_arg_int(int arg_id, int64 d);
  // Bulk processing of multiple scalar int arguments at the same time.
  // See 'set_arg_float' documentation for details.
  void set_args_int(const std::vector<int> &args_id, const std::vector<int64> &vec);
  // Bulk processing of multiple scalar uint arguments at the same time.
  // See 'set_arg_float' documentation for details.
  void set_arg_uint(int arg_id, uint64 d);
  void set_args_uint(const std::vector<int> &args_id, const std::vector<uint64> &vec);

  void set_array_runtime_size(int arg_id, uint64 size);

  void set_array_device_allocation_type(int arg_id, DevAllocType usage);

  template <typename T>
  void set_arg(int arg_id, T v);

  // The following two functions can be used to set struct args and primitive
  // args. The first element of `arg_indices` is the index of the argument. The
  // rest of the elements are the index of the field in each depth of the nested
  // struct.

  template <typename T, std::size_t N>
  void set_struct_arg_impl(const std::array<int, N> &arg_indices, T v);

  template <typename T, std::size_t N>
  void set_struct_arg(const std::array<int, N> &arg_indices, T v);

  template <typename T>
  void set_struct_arg(const std::vector<int> &arg_indices, T v);

  void set_ndarray_ptrs(int arg_id, uint64 data_ptr, uint64 grad_ptr);
  // Same as `set_ndarray_ptrs`, but also mirrors the resolved host pointer into `array_ptrs` so the adstack
  // size-expression evaluator can dereference it. Call only from launchers where `data_ptr`/`grad_ptr` is a
  // real host-accessible address (CPU); device-only launchers (SPIR-V / CUDA / AMDGPU) must use the plain
  // `set_ndarray_ptrs`.
  void set_host_accessible_ndarray_ptrs(int arg_id, uint64 data_ptr, uint64 grad_ptr);

  template <typename T>
  T get_arg(const std::vector<int> &i);

  template <typename T>
  T get_struct_arg(std::vector<int> arg_indices);

  // Host-only counterpart of `get_struct_arg`: always reads from the launcher-owned `arg_buffer_` host backing
  // store instead of `RuntimeContext::arg_buffer`. Needed because CUDA / AMDGPU launchers swap
  // `ctx_->arg_buffer` to a device pointer before handing the context to the adstack sizer encoder; a plain
  // `get_struct_arg<T>` at that point would dereference device memory from the host. Call this from host-side
  // evaluators (e.g. `encode_adstack_size_expr_device_bytecode`) that need to peek scalar slots without
  // assuming the kernel-facing `RuntimeContext::arg_buffer` still points at host memory.
  template <typename T>
  T get_struct_arg_host(std::vector<int> arg_indices);

  template <typename T>
  T get_ret(int arg_id);

  void set_arg_external_array_with_shape(int arg_id,
                                         uintptr_t ptr,
                                         uint64 size,
                                         const std::vector<int64> &shape,
                                         uintptr_t grad_ptr = 0);

  void set_arg_ndarray_impl(int arg_id,
                            intptr_t devalloc_ptr,
                            const std::vector<int> &shape,
                            intptr_t devalloc_ptr_grad = 0);
  void set_arg_ndarray(int arg_id, const Ndarray &arr);
  // Bulk processing of multiple individual Taichi NDarray arguments (without
  // any associated gradient) at the same time.
  // See 'set_arg_float' for details.
  void set_args_ndarray(const std::vector<int> &args_id, const std::vector<Ndarray *> &arrs);
  void set_arg_ndarray_with_grad(int arg_id, const Ndarray &arr, const Ndarray &arr_grad);
  // Bulk processing of multiple individual Taichi NDarray arguments (along
  // with associated gradient) at the same time.
  // See 'set_arg_float' for details.
  void set_args_ndarray_with_grad(const std::vector<int> &args_id,
                                  const std::vector<Ndarray *> &arrs,
                                  const std::vector<Ndarray *> &arrs_grad);

  void set_arg_matrix(int arg_id, const Matrix &matrix);

  TypedConstant fetch_ret(const std::vector<int> &index);
  float64 get_struct_ret_float(const std::vector<int> &index);
  int64 get_struct_ret_int(const std::vector<int> &index);
  uint64 get_struct_ret_uint(const std::vector<int> &index);

  RuntimeContext &get_context();

 private:
  TypedConstant fetch_ret_impl(int offset, const Type *dt);
  CallableBase *kernel_;
  std::unique_ptr<RuntimeContext> owned_ctx_;
  // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
  // that the caller passes a RuntimeContext pointer externally. In that case,
  // |owned_ctx_| will be nullptr.
  // Invariant: |ctx_| will never be nullptr.
  RuntimeContext *ctx_;
  std::unique_ptr<char[]> arg_buffer_;
  std::unique_ptr<char[]> result_buffer_;
  const StructType *ret_type_;

 public:
  size_t arg_buffer_size{0};
  const StructType *args_type{nullptr};
  size_t result_buffer_size{0};
  bool use_graph{false};
  // Level table for nested `graph_do_while`, indexed by level id (empty if the kernel has no graph_do_while loop).
  // Populated from Python at launch; flag_dev_ptr filled in by the backend's ndarray-resolution loop. Replaces the old
  // single-loop scalars (arg id + flag ptr).
  std::vector<GraphDoWhileLevel> graph_do_while_levels;

  // True if this kernel has at least one graph_do_while loop.
  bool has_graph_do_while() const {
    return !graph_do_while_levels.empty();
  }

  // Append a graph_do_while level (called from Python at launch, in level-id order). `cond_arg_id` is the resolved C++
  // arg index of the condition ndarray; `parent_id` is the enclosing level id or -1.
  void add_graph_do_while_level(int cond_arg_id, int parent_id) {
    graph_do_while_levels.push_back(GraphDoWhileLevel{cond_arg_id, parent_id, nullptr});
  }

  // Resolve the flag device pointer for any level whose condition arg matches `arg_id`. Called by each backend's
  // per-arg ndarray-resolution loop. (A given ndarray could in principle drive more than one level, so we set all
  // matches rather than break.)
  void resolve_graph_do_while_flag(int arg_id, void *flag_dev_ptr) {
    for (auto &level : graph_do_while_levels) {
      if (level.cond_arg_id == arg_id) {
        level.flag_dev_ptr = flag_dev_ptr;
      }
    }
  }

  // Note that I've tried to group `array_runtime_size` and
  // `is_device_allocations` into a small struct. However, it caused some test
  // cases to stuck.

  // `array_runtime_size` records the runtime size of the
  // corresponding array arguments.
  std::unordered_map<int, uint64> array_runtime_sizes;
  // `device_allocation_type` is set iff i-th arg is a `DeviceAllocation*`,
  // otherwise it is set to DevAllocType::kNone
  std::unordered_map<int, DevAllocType> device_allocation_type;

  std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher> array_ptrs;
  // Per-arg ndarray shape, populated by `set_arg_external_array_with_shape` / `set_arg_ndarray*`. Mirrors what
  // is already encoded into `arg_buffer_` via `set_struct_arg(std::array{arg_id, 0, axis}, shape[axis])`, but
  // exposed as a flat vector here so the diagnose path (`Program::capture_diagnose_snapshot`) does not have to
  // walk `args_type` element offsets. The args-type walk would emit spurious "Cannot treat as TensorType"
  // diagnostics on out-of-rank axis lookups; mirroring shapes once at set-time is both cheaper and quieter.
  std::unordered_map<int, std::vector<int>> ndarray_shapes;
};

}  // namespace quadrants::lang
