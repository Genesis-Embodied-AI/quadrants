#pragma once

#include <optional>
#include <string>
#include <vector>

#include "quadrants/ir/offloaded_task_type.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {

class Kernel;
class SNode;

namespace spirv {

/**
 * Per offloaded task attributes.
 */
struct TaskAttributes {
  enum class BufferType {
    Root,
    GlobalTmps,
    Args,
    Rets,
    ListGen,
    ExtArr,
    AdStackOverflow,
    AdStackHeapFloat,
    AdStackHeapInt,
  };

  struct BufferInfo {
    BufferType type;
    int root_id{-1};  // only used if type==Root or type==ExtArr
    // For type==ExtArr only: true selects the gradient mirror of the ndarray argument instead of its data buffer.
    // Reverse-mode AD kernels need a distinct StorageBuffer binding so data and grad end up in different device
    // allocations on backends without physical_storage_buffer.
    bool is_grad{false};

    BufferInfo() = default;

    // NOLINTNEXTLINE(google-explicit-constructor)
    BufferInfo(BufferType buffer_type) : type(buffer_type) {
    }

    BufferInfo(BufferType buffer_type, int root_buffer_id, bool is_grad = false)
        : type(buffer_type), root_id(root_buffer_id), is_grad(is_grad) {
    }

    bool operator==(const BufferInfo &other) const {
      if (type != other.type) {
        return false;
      }
      if (type == BufferType::ExtArr && is_grad != other.is_grad) {
        return false;
      }
      if (type == BufferType::Root || type == BufferType::ExtArr) {
        return root_id == other.root_id;
      }
      return true;
    }

    QD_IO_DEF(type, root_id, is_grad);
  };

  struct BufferInfoHasher {
    std::size_t operator()(const BufferInfo &buf) const {
      using std::hash;
      using std::size_t;
      using std::string;

      size_t hash_result = hash<BufferType>()(buf.type);
      hash_result ^= buf.root_id;
      // Mix `is_grad` only for ExtArr: operator== only looks at `is_grad` when type == ExtArr, so doing the
      // same here keeps the hasher consistent with equality. Hashing `is_grad` on other BufferTypes would
      // split equal keys across buckets and violate the unordered-container invariant.
      // 0x9e3779b9 is the `hash_combine` golden-ratio fractional constant (same one boost::hash_combine uses).
      // Preferred over `(size_t)is_grad << 16` because root_id values near 0x10000 would collide with a shifted
      // is_grad bit; the full-word constant keeps the two axes independent.
      if (buf.type == BufferType::ExtArr && buf.is_grad) {
        hash_result ^= std::size_t(0x9e3779b9ULL);
      }
      return hash_result;
    }
  };

  struct BufferBind {
    BufferInfo buffer;
    int binding{0};

    std::string debug_string() const;

    QD_IO_DEF(buffer, binding);
  };

  std::string name;
  std::string source_path;
  // Total number of threads to launch (i.e. threads per grid). Note that this
  // is only advisory, because eventually this number is also determined by the
  // runtime config. This works because grid strided loop is supported.
  int advisory_total_num_threads{0};
  int advisory_num_threads_per_group{0};

  OffloadedTaskType task_type;

  struct RangeForAttributes {
    // |begin| has different meanings depending on |const_begin|:
    // * true : It is the left boundary of the loop known at compile time.
    // * false: It is the offset of the begin in the global tmps buffer.
    //
    // Same applies to |end|.
    size_t begin{0};
    size_t end{0};
    bool const_begin{true};
    bool const_end{true};

    inline bool const_range() const {
      return (const_begin && const_end);
    }

    // When the range end is non-const and the IR encodes it as a product of one or more ndarray-shape lookups (a
    // common `qd.ndrange(arr.shape[...], ...)` pattern), the codegen extracts each `ExternalTensorShapeAlongAxisStmt`
    // into this list. At launch time, host-side `LaunchContextBuilder` already has every ndarray's shape in
    // `array_ptrs` / struct args, so the actual iteration bound is `product over refs of arg[arg_id].shape[axis]`.
    // The runtime uses that as a tight cap on `advisory_total_num_threads` to avoid oversizing the per-thread
    // adstack heap (otherwise `kMaxNumThreadsGridStrideLoop` defaults to 131072 for a B=1 workload and the heap
    // allocation requests multi-GB that exceeds Metal's `maxBufferLength`). Empty means the end expression could
    // not be simplified to a pure product of shape lookups; fall back to the advisory thread count in that case.
    struct ArgShapeRef {
      std::vector<int> arg_id;
      int axis{0};
      QD_IO_DEF(arg_id, axis);
    };
    std::vector<ArgShapeRef> end_shape_product;

    QD_IO_DEF(begin, end, const_begin, const_end, end_shape_product);
  };
  std::vector<BufferBind> buffer_binds;
  // Only valid when |task_type| is range_for.
  std::optional<RangeForAttributes> range_for_attribs;

  // Per-thread stride, in f32 elements, of the f32-typed heap-backed adstack slice used by this task, bound as
  // BufferType::AdStackHeapFloat. Zero when the task has no f32 adstack. The runtime multiplies this by the
  // dispatched invocation count to size the shared adstack buffer.
  uint32_t ad_stack_heap_per_thread_stride_float{0};
  // Per-thread stride, in i32 elements, of the int-typed heap-backed adstack slice used by this task, bound as
  // BufferType::AdStackHeapInt. Backs both i32 and u1 adstacks (u1 is stored as i32, matching the existing
  // Function-scope path). Zero when the task has no non-f32 adstack.
  uint32_t ad_stack_heap_per_thread_stride_int{0};

  static std::string buffers_name(BufferInfo b);

  std::string debug_string() const;

  QD_IO_DEF(name,
            advisory_total_num_threads,
            advisory_num_threads_per_group,
            task_type,
            buffer_binds,
            range_for_attribs,
            ad_stack_heap_per_thread_stride_float,
            ad_stack_heap_per_thread_stride_int);
};

/**
 * This class contains the attributes descriptors for both the input args and
 * the return values of a Quadrants kernel.
 *
 * Note that all SPIRV tasks (shaders) belonging to the same Quadrants kernel
 * will share the same kernel args (i.e. they use the same device buffer for
 * input args and return values). This is because kernel arguments is a
 * Quadrants-level concept.
 *
 * Memory layout
 *
 * /---- input args ----\/---- ret vals -----\/-- extra args --\
 * +----------+---------+----------+---------+-----------------+
 * |  scalar  |  array  |  scalar  |  array  |      scalar     |
 * +----------+---------+----------+---------+-----------------+
 */
class KernelContextAttributes {
 private:
  /**
   * Attributes that are shared by the input arg and the return value.
   */
  struct AttribsBase {
    std::string name;
    // For scalar arg, this is max(stride(dt), 4)
    // For array arg, this is #elements * max(stride(dt), 4)
    // Unit: byte
    size_t stride{0};
    // Offset in the context buffer
    size_t offset_in_mem{0};
    PrimitiveTypeID dtype{PrimitiveTypeID::unknown};
    bool is_array{false};
    std::vector<int> element_shape;
    std::size_t field_dim{0};
    ParameterType ptype{ParameterType::kUnknown};

    QD_IO_DEF(name, stride, offset_in_mem, dtype, is_array, element_shape, field_dim, ptype);
  };

 public:
  /**
   * This is mostly the same as Kernel::Arg, with device specific attributes.
   */
  struct ArgAttributes : public AttribsBase {
    // Indices of the arg value in the host `Context`.
    std::vector<int> indices;

    QD_IO_DEF(name, stride, offset_in_mem, indices, dtype, is_array, element_shape, field_dim, ptype);
  };

  /**
   * This is mostly the same as Kernel::Ret, with device specific attributes.
   */
  struct RetAttributes : public AttribsBase {
    // Index of the return value in the host `Context`.
    int index{-1};

    QD_IO_DEF(name, stride, offset_in_mem, index, dtype, is_array, element_shape, field_dim, ptype);
  };

  KernelContextAttributes() = default;
  explicit KernelContextAttributes(const Kernel &kernel, const DeviceCapabilityConfig *caps);

  /**
   * Whether this kernel has any argument
   */
  inline bool has_args() const {
    return !arg_attribs_vec_.empty();
  }

  inline const std::vector<std::pair<std::vector<int>, ArgAttributes>> &args() const {
    return arg_attribs_vec_;
  }

  inline const ArgAttributes &arg_at(const std::vector<int> &indices) const {
    for (const auto &element : arg_attribs_vec_) {
      if (element.first == indices) {
        return element.second;
      }
    }
    QD_ERROR(fmt::format("Unexpected error: ArgAttributes with indices ({}) not found.", fmt::join(indices, ", ")));
    return arg_attribs_vec_[0].second;
  }

  /**
   * Whether this kernel has any return value
   */
  inline bool has_rets() const {
    return !ret_attribs_vec_.empty();
  }

  inline const std::vector<RetAttributes> &rets() const {
    return ret_attribs_vec_;
  }

  /**
   * Whether this kernel has either arguments or return values.
   */
  inline bool empty() const {
    return !(has_args() || has_rets());
  }

  /**
   * Number of bytes needed by all the arguments.
   */
  inline size_t args_bytes() const {
    return args_bytes_;
  }

  /**
   * Number of bytes needed by all the return values.
   */
  inline size_t rets_bytes() const {
    return rets_bytes_;
  }

  /**
   * The type of the struct that contains all the arguments.
   */
  inline const lang::StructType *args_type() const {
    return args_type_;
  }

  /**
   * The type of the struct that contains all the return values.
   */
  inline const lang::StructType *rets_type() const {
    return rets_type_;
  }

  std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>> arr_access;

  QD_IO_DEF(arg_attribs_vec_, ret_attribs_vec_, args_bytes_, rets_bytes_, arr_access, args_type_, rets_type_);

 private:
  std::vector<std::pair<std::vector<int>, ArgAttributes>> arg_attribs_vec_;
  std::vector<RetAttributes> ret_attribs_vec_;

  size_t args_bytes_{0};
  size_t rets_bytes_{0};

  const lang::StructType *args_type_{nullptr};
  const lang::StructType *rets_type_{nullptr};
};

/**
 * Groups all the device kernels generated from a single ti.kernel.
 */
struct QuadrantsKernelAttributes {
  // Quadrants kernel name
  std::string name;
  // Is this kernel for evaluating the constant fold result?
  bool is_jit_evaluator{false};
  // Attributes of all the tasks produced from this single Quadrants kernel.
  std::vector<TaskAttributes> tasks_attribs;

  KernelContextAttributes ctx_attribs;

  QD_IO_DEF(name, is_jit_evaluator, tasks_attribs, ctx_attribs);
};

}  // namespace spirv
}  // namespace quadrants::lang
