#pragma once

#include <optional>
#include <string>
#include <vector>

#include "quadrants/ir/adstack_size_expr.h"
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
    // Per-dispatch StorageBuffer holding the runtime-evaluated adstack geometry (`stride_float`,
    // `stride_int`, then `(offset, max_size)` pairs per alloca in pre-scan order). Populated by the
    // host launcher before each dispatch from the symbolic `SizeExpr` trees captured in
    // `TaskAttributes::ad_stack`; the shader reads these u32 entries at every
    // `AdStackAllocaStmt` / `AdStackPushStmt` / `AdStackLoadTopStmt` site so the per-thread slice
    // layout tightens to the actual field state at each launch. Zero-sized and unbound when a
    // task declares no adstacks.
    AdStackMetadata,
    // Per-dispatch StorageBuffer holding a single u32 atomic counter used to lazily claim
    // per-thread heap rows. Threads that reach an AdStackPushStmt (or LoadTop / LoadTopAdj)
    // atomicAdd this counter and use the returned index as their row id; threads that never enter
    // a push site never increment the counter and consume zero heap rows. Host clears the slot to
    // 0 before each dispatch and reads it back after to drive the grow-and-retry path on the
    // float / int heap allocations. Zero-sized and unbound when the task declares no adstacks or
    // when the codegen falls back to the eager invoc-id-based row layout (e.g. when the LCA-of-
    // pushes pre-pass cannot place a single dominator claim site).
    AdStackRowCounter,
    // Per-kernel StorageBuffer holding the static-IR-bound row capacity per task (`uint[num_tasks_in_kernel]`).
    // Populated by the host after the bound-reducer dispatch (see `runtime/gfx/adstack_bound_reducer_launch.cpp`):
    // for each task with a captured `bound_expr`, slot `task_id_in_kernel` carries the exact count of
    // threads the reducer observed passing the gate; for every other task the host writes UINT32_MAX so the
    // bounds check below is inert. The main-task SPIR-V loads this slot at the Lowest Common Ancestor (LCA) block
    // claim site immediately after the OpAtomicIAdd that produces `claimed_row` and OpAtomicUMax-signals
    // UINT32_MAX into AdStackOverflow when `claimed_row >= capacity`. The expected behaviour is "this signal
    // never fires on legitimate workloads" because the reducer count is exact by construction; if it does fire,
    // it indicates a reducer / main divergence (an internal bug, not user-recoverable), and `synchronize()`
    // surfaces it as a clear actionable error rather than letting it silently corrupt gradients via OOB writes.
    AdStackBoundRowCapacity,
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

  // Per-adstack compile-time metadata for this task, one entry per `AdStackAllocaStmt` indexed by its
  // pre-scan order (the same ordering the shader uses as `stack_id`). `heap_kind` selects which of the
  // `AdStackHeapFloat` / `AdStackHeapInt` buffers backs the alloca; `offset_in_elems_compile_time` is
  // the prefix-sum offset produced from `max_size_compile_time` (used when `size_expr` is empty, i.e.
  // after an offline-cache load where the symbolic tree was not serialised); `size_expr` is the flat
  // post-order symbolic upper-bound tree captured by the `determine_ad_stack_size` pre-pass, evaluated
  // at each launch to publish the runtime-tight `offset` and `max_size` into the `AdStackMetadata`
  // buffer. An empty `size_expr::nodes` means "no symbolic bound captured, use
  // `max_size_compile_time`"; the runtime falls back to that value on the cache-hit path.
  struct AdStackAllocaAttribs {
    enum class HeapKind : int32_t { Float = 0, Int = 1 };
    HeapKind heap_kind{HeapKind::Float};
    uint32_t offset_in_elems_compile_time{0};
    uint32_t max_size_compile_time{0};
    SerializedSizeExpr size_expr{};
    QD_IO_DEF(heap_kind, offset_in_elems_compile_time, max_size_compile_time, size_expr);
  };
  // Captured upper bound on the per-task LCA-block-reaching thread count, derived at codegen time
  // by walking the LCA dominator chain and pattern-matching the gating condition. When set, the
  // runtime dispatches a generic reducer kernel before the main task to evaluate the captured
  // predicate over the bound iteration range; the resulting count is then used to size the
  // AdStackHeapFloat / AdStackHeapInt allocations exactly. When `nullopt` (the gate did not match
  // a recognized grammar, or the LCA pre-pass placed the LCA at the task body root with no gate
  // above it), the runtime falls back to the dispatched-threads worst-case sizing - no behavior
  // change versus a kernel without this metadata.
  //
  // The grammar (recognized by `find_adstack_bound_expr`): exactly one `IfStmt` gate above the
  // LCA, condition shaped as `BinaryOp(cmp, GlobalLoadStmt(field[I]), ConstStmt(literal))` where
  // `cmp` is in `{<, <=, >, >=, ==, !=}` and `I` is a loop index of the dispatch range. the compound-predicate path
  // extends to compound predicates over `BinaryOp(land, ...)` / `BinaryOp(lor, ...)` trees of the
  // leaf shape; that path leaves the leaf-only fields below empty and uses the post-order
  // node tree instead.
  struct StaticBoundExpr {
    // Comparison op (stored as int instead of `BinaryOpType` to keep this header dependency-light;
    // the IR pass and the runtime reducer kernel agree on the encoding via cast through
    // `BinaryOpType`).
    int cmp_op{0};

    // Literal threshold from the gate's right-hand `ConstStmt`. The active union member is
    // selected by the GlobalLoad result's primitive type the IR pass observed; the reducer kernel
    // bitcasts the right one based on `field_dtype_is_float` at dispatch time.
    bool field_dtype_is_float{true};
    float literal_f32{0.0f};
    int32_t literal_i32{0};

    // True when the LCA enters on the gate condition holding (typical `if cmp:` shape); false when
    // the LCA sits inside the `else` branch (`if cmp: else: <gate>`). The reducer flips the
    // predicate accordingly so the captured count always matches threads that reach LCA.
    bool polarity{true};

    // Field source. SNode-backed fields (`qd.field(...)` placed under `qd.root.dense(...)`) are identified
    // by the leaf snode's global id; ndarray-backed kernel arguments (`qd.ndarray(...)`) are identified by
    // the `arg_id` list pointing into the kernel arg buffer. The IR pass accepts both because sparse-grid
    // workloads tend to gate on SNode-backed scalar fields while smaller test repros tend to use ndarrays.
    enum class FieldSourceKind : int32_t { SNode = 0, NdArray = 1 };
    FieldSourceKind field_source_kind{FieldSourceKind::SNode};
    int snode_id{-1};
    std::vector<int> ndarray_arg_id;

    // SNode-source extras. The IR pattern matcher fills these for `SNode`-backed gates by walking the snode
    // descriptor chain from the leaf up to root and combining the per-level `mem_offset_in_parent_cell` /
    // `cell_stride` values into a flat (base, stride) pair the reducer shader can use without re-emitting the
    // SNode-tree lookup chain at dispatch time. `snode_root_id` selects the right root buffer to bind on the
    // reducer dispatch (a kernel may have multiple roots; the snode-descriptor map is keyed per-root). Set to
    // -1 / 0 for ndarray-backed gates.
    int snode_root_id{-1};
    uint32_t snode_byte_base_offset{0};
    uint32_t snode_byte_cell_stride{0};
    uint32_t snode_iter_count{0};

    QD_IO_DEF(cmp_op,
              field_dtype_is_float,
              literal_f32,
              literal_i32,
              polarity,
              field_source_kind,
              snode_id,
              ndarray_arg_id,
              snode_root_id,
              snode_byte_base_offset,
              snode_byte_cell_stride,
              snode_iter_count);
  };

  struct AdStackSizingAttribs {
    // Compile-time-derived per-thread strides in elements of each heap's element type. The runtime
    // recomputes these when any alloca's `size_expr` evaluates dynamically; the compile-time values
    // serve both as the offline-cache-serialised fallback (empty `size_expr` on every alloca) and as
    // the upper bound for heap-buffer growth when no adstacks are declared (kept at zero). Writing
    // the final per-launch strides into the metadata buffer slots (0 and 1) is done by the host
    // launcher regardless of whether any alloca's bound was dynamic.
    uint32_t per_thread_stride_float_compile_time{0};
    uint32_t per_thread_stride_int_compile_time{0};
    std::vector<AdStackAllocaAttribs> allocas;
    std::optional<StaticBoundExpr> bound_expr;
    QD_IO_DEF(per_thread_stride_float_compile_time, per_thread_stride_int_compile_time, allocas, bound_expr);
  };
  AdStackSizingAttribs ad_stack;

  static std::string buffers_name(BufferInfo b);

  std::string debug_string() const;

  QD_IO_DEF(name,
            advisory_total_num_threads,
            advisory_num_threads_per_group,
            task_type,
            buffer_binds,
            range_for_attribs,
            ad_stack);
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

  // Per-arg access bits restricted to the `.grad` slot (`ExternalPtrStmt::is_grad == true` references). Indexed
  // by the same `std::vector<int>` arg-id key as `arr_access` and filled in parallel with it. The
  // `GfxRuntime::launch_kernel` blit path gates the host->device grad mirror on this; a forward kernel that
  // never reads / writes `.grad` has every entry at `NONE` and skips the map+memcpy+unmap per grad-bearing arg.
  std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>> grad_arr_access;

  QD_IO_DEF(arg_attribs_vec_,
            ret_attribs_vec_,
            args_bytes_,
            rets_bytes_,
            arr_access,
            grad_arr_access,
            args_type_,
            rets_type_);

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
