#include "quadrants/codegen/spirv/spirv_codegen.h"

#include <string>
#include <string_view>
#include <vector>
#include <variant>
#include <filesystem>

#include "spirv/unified1/GLSL.std.450.h"
#include "quadrants/codegen/codegen_utils.h"
#include "quadrants/program/program.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/ir.h"
#include "quadrants/util/line_appender.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "quadrants/codegen/spirv/detail/spirv_codegen.h"
#include "quadrants/codegen/spirv/spirv_shared_array_retyping.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/snode.h"
#include "quadrants/math/arithmetic.h"
#include "quadrants/codegen/ir_dump.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace quadrants::lang {
namespace spirv {

constexpr char kRootBufferName[] = "root_buffer";
constexpr char kGlobalTmpsBufferName[] = "global_tmps_buffer";
constexpr char kArgsBufferName[] = "args_buffer";
constexpr char kRetBufferName[] = "ret_buffer";
constexpr char kListgenBufferName[] = "listgen_buffer";
constexpr char kExtArrBufferName[] = "ext_arr_buffer";
constexpr char kAdStackOverflowBufferName[] = "adstack_overflow_buffer";
constexpr char kAdStackRowCounterBufferName[] = "adstack_row_counter_buffer";
constexpr char kAdStackBoundRowCapacityBufferName[] = "adstack_bound_row_capacity_buffer";
constexpr char kAdStackTaskRegistryIdBufferName[] = "adstack_task_registry_id_buffer";
constexpr char kAdStackHeapFloatBufferName[] = "adstack_heap_float_buffer";
constexpr char kAdStackHeapIntBufferName[] = "adstack_heap_int_buffer";
constexpr char kAdStackMetadataBufferName[] = "adstack_metadata_buffer";

constexpr int kMaxNumThreadsGridStrideLoop = 65536 * 2;

using BufferType = TaskAttributes::BufferType;
using namespace detail;

std::string buffer_instance_name(BufferInfo b) {
  // https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Syntax
  switch (b.type) {
    case BufferType::Root:
      return std::string(kRootBufferName) + "_" + std::to_string(b.root_id);
    case BufferType::GlobalTmps:
      return kGlobalTmpsBufferName;
    case BufferType::Args:
      return kArgsBufferName;
    case BufferType::Rets:
      return kRetBufferName;
    case BufferType::ListGen:
      return kListgenBufferName;
    case BufferType::ExtArr:
      return std::string(kExtArrBufferName) + "_" + std::to_string(b.root_id) + (b.is_grad ? "_grad" : "");
    case BufferType::AdStackOverflow:
      return kAdStackOverflowBufferName;
    case BufferType::AdStackRowCounter:
      return kAdStackRowCounterBufferName;
    case BufferType::AdStackBoundRowCapacity:
      return kAdStackBoundRowCapacityBufferName;
    case BufferType::AdStackTaskRegistryId:
      return kAdStackTaskRegistryIdBufferName;
    case BufferType::AdStackHeapFloat:
      return kAdStackHeapFloatBufferName;
    case BufferType::AdStackHeapInt:
      return kAdStackHeapIntBufferName;
    case BufferType::AdStackMetadata:
      return kAdStackMetadataBufferName;
    default:
      QD_NOT_IMPLEMENTED;
      break;
  }
  return {};
}

TaskCodegen::TaskCodegen(const Params &params)
    : arch_(params.arch),
      caps_(params.caps),
      compile_config_(params.compile_config),
      task_id_in_kernel_(params.task_id_in_kernel),
      task_ir_(params.task_ir),
      compiled_structs_(params.compiled_structs),
      ctx_attribs_(params.ctx_attribs),
      task_name_(params.task_ir->loop_name.empty()
                     ? fmt::format("{}_t{:02d}", params.ti_kernel_name, params.task_id_in_kernel)
                     : fmt::format("{}_t{:02d}_{}",
                                   params.ti_kernel_name,
                                   params.task_id_in_kernel,
                                   params.task_ir->loop_name)) {
  allow_undefined_visitor = true;
  invoke_default_visitor = true;

  fill_snode_to_root();
  ir_ = std::make_shared<spirv::IRBuilder>(arch_, caps_);
  // Workaround for Metal/MoltenVK shader compiler bug: the compiler
  // incorrectly hoists storage buffer loads out of loops (LICM), causing
  // stale reads when a buffer is written and re-read within the same loop.
  // Marking buffer accesses as Volatile prevents this optimization.
  use_volatile_buffer_access_ = (arch_ == Arch::metal);
}

void TaskCodegen::fill_snode_to_root() {
  for (int root = 0; root < compiled_structs_.size(); ++root) {
    for (auto &[node_id, node] : compiled_structs_[root].snode_descriptors) {
      snode_to_root_[node_id] = root;
    }
  }
}

// Replace the wild '%' in the format string with "%%".
std::string TaskCodegen::sanitize_format_string(std::string const &str) {
  std::string sanitized_str;
  for (char c : str) {
    if (c == '%') {
      sanitized_str += "%%";
    } else {
      sanitized_str += c;
    }
  }
  return sanitized_str;
}

struct Result {
  std::vector<uint32_t> spirv_code;
  TaskAttributes task_attribs;
  std::unordered_map<std::vector<int>, irpass::ExternalPtrAccess, hashing::Hasher<std::vector<int>>> arr_access;
};

TaskCodegen::Result TaskCodegen::run() {
  ir_->init_header();
  kernel_function_ = ir_->new_function();  // void main();
  ir_->debug_name(spv::OpName, kernel_function_, "main");
  scan_shared_atomic_allocs(task_ir_->body.get(), shared_float_allocas_with_atomic_rmw_);

  // Run the shared static-adstack analysis over the task body. Returns the LCA of every f32 push/load-top site, the set
  // of autodiff-bootstrap const-init pushes the codegen must skip the slot store for, the per-thread strides, and an
  // optional `StaticBoundExpr` capturing the gating predicate when the LCA-to-root chain has a single recognized gate.
  // The SNode descriptor resolver below turns the SPIR-V backend's `compiled_structs_` / `snode_to_root_` state into
  // the generic `SNodeFieldDescriptor` the analysis consumes; ndarray-backed gates are recognized without the resolver.
  auto snode_descriptor_resolver = [this](const SNode *leaf,
                                          const SNode *dense) -> std::optional<SNodeFieldDescriptor> {
    if (leaf == nullptr || dense == nullptr || dense->parent == nullptr) {
      return std::nullopt;
    }
    auto root_it = snode_to_root_.find(dense->parent->id);
    if (root_it == snode_to_root_.end()) {
      return std::nullopt;
    }
    const int root_id = root_it->second;
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    auto leaf_desc_it = snode_descs.find(leaf->id);
    auto dense_desc_it = snode_descs.find(dense->id);
    if (leaf_desc_it == snode_descs.end() || dense_desc_it == snode_descs.end()) {
      return std::nullopt;
    }
    SNodeFieldDescriptor desc;
    desc.root_id = root_id;
    // Combined byte offset: dense's offset within its single root cell plus the leaf's offset within the dense's
    // per-cell layout. Both come from the snode descriptor's compile-time prefix-sum so the captured value is stable
    // across launches.
    desc.byte_base_offset = static_cast<uint32_t>(dense_desc_it->second.mem_offset_in_parent_cell +
                                                  leaf_desc_it->second.mem_offset_in_parent_cell);
    desc.byte_cell_stride = static_cast<uint32_t>(dense_desc_it->second.cell_stride);
    desc.iter_count = static_cast<uint32_t>(dense_desc_it->second.total_num_cells_from_root);
    return desc;
  };
  auto adstack_analysis = analyze_adstack_static_bounds(task_ir_, snode_descriptor_resolver,
                                                        compile_config_->ad_stack_sparse_threshold_bytes);
  ad_stack_heap_per_thread_stride_float_ = adstack_analysis.per_thread_stride_float;
  ad_stack_heap_per_thread_stride_int_ = adstack_analysis.per_thread_stride_int;
  num_ad_stacks_ = adstack_analysis.num_ad_stacks;
  ad_stack_lca_block_float_ = adstack_analysis.lca_block_float;
  ad_stack_bootstrap_pushes_ = std::move(adstack_analysis.bootstrap_pushes);
  if (adstack_analysis.bound_expr.has_value()) {
    task_attribs_.ad_stack.bound_expr = *adstack_analysis.bound_expr;
  }

  if (task_ir_->task_type == OffloadedTaskType::serial) {
    generate_serial_kernel(task_ir_);
  } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
    // struct_for is automatically lowered to ranged_for for dense snodes
    generate_range_for_kernel(task_ir_);
  } else if (task_ir_->task_type == OffloadedTaskType::struct_for) {
    generate_struct_for_kernel(task_ir_);
  } else {
    QD_ERROR("Unsupported offload type={} on SPIR-V codegen", task_ir_->task_name());
  }
  // Headers need global information, so it has to be delayed after visiting
  // the task IR.
  emit_headers();

  task_attribs_.ad_stack.per_thread_stride_float_compile_time = ad_stack_heap_per_thread_stride_float_;
  task_attribs_.ad_stack.per_thread_stride_int_compile_time = ad_stack_heap_per_thread_stride_int_;
  // recognize `MaxOverRange` nodes the runtime can reduce in parallel via the dedicated max-reducer dispatch instead of
  // letting the per-thread sizer enumerate. Indexing matches `task_attribs_.ad_stack.allocas` (each entry's `size_expr`
  // is the per-stack tree captured above).
  {
    std::vector<SerializedSizeExpr> per_stack_size_exprs;
    per_stack_size_exprs.reserve(task_attribs_.ad_stack.allocas.size());
    for (const auto &a : task_attribs_.ad_stack.allocas) {
      per_stack_size_exprs.push_back(a.size_expr);
    }
    task_attribs_.ad_stack.max_reducer_specs = recognize_adstack_max_reducer_specs(per_stack_size_exprs);
  }

  // Snodes the task body mutates (any `GlobalStore` or `AtomicOp` whose dest resolves to a
  // `GlobalPtrStmt`). Persisted on `task_attribs_.snode_writes` so the SPIR-V launcher can bump
  // `Program::snode_write_gen_` for each id on every `launch_kernel` call - that is the precise signal
  // the per-task adstack metadata cache uses to invalidate when a prior kernel may have changed a
  // value an enclosing `size_expr::FieldLoad` reads. Stored as raw int ids (not `SNode *`) so the
  // field round-trips through the offline cache without resolving the pointer at serialise time.
  auto snode_rw = irpass::analysis::gather_snode_read_writes(task_ir_);
  task_attribs_.snode_writes.reserve(snode_rw.second.size());
  for (auto *s : snode_rw.second) {
    if (s != nullptr) {
      task_attribs_.snode_writes.push_back(s->id);
    }
  }
  std::sort(task_attribs_.snode_writes.begin(), task_attribs_.snode_writes.end());
  task_attribs_.snode_writes.erase(std::unique(task_attribs_.snode_writes.begin(), task_attribs_.snode_writes.end()),
                                   task_attribs_.snode_writes.end());

  Result res;
  res.spirv_code = ir_->finalize();
  res.task_attribs = std::move(task_attribs_);
  res.arr_access = irpass::detect_external_ptr_access_in_task(task_ir_);
  res.grad_arr_access = irpass::detect_external_ptr_grad_access_in_task(task_ir_);

  return res;
}

void TaskCodegen::visit(OffloadedStmt *) {
  QD_ERROR("This codegen is supposed to deal with one offloaded task");
}

void TaskCodegen::visit(Block *stmt) {
  // Sparse adstack heap: when codegen enters the float Lowest Common Ancestor (LCA) block of every f32-typed
  // AdStackPushStmt / AdStackLoadTopStmt / AdStackLoadTopAdjStmt in this task, atomically claim a heap row id for this
  // thread and store it into the Function-scope `ad_stack_row_id_var_float_`. The claim runs exactly once per thread
  // per task: every thread that reaches a float push / load-top must first pass through this block (by definition of
  // LCA), and a thread that does not pass through this block also never reaches a float push or load-top, so the
  // unclaimed row_id_var (UINT32_MAX) is observable only at sites that are guaranteed not to execute. The store happens
  // BEFORE any of this block's statements are codegen'd so all descendant push / load-top sites observe the claimed
  // value. Both the `row_id_var` allocation and its UINT32_MAX-initialisation live on the same block-entry hook so that
  // when the float LCA is the task body root (typical for kernels without a predicate gating all f32 pushes), the init
  // store dominates the atomic claim. `alloca_variable` hoists the OpVariable to the SPIR-V function entry block
  // regardless of where it is called from, but the OpStore lands here in the LCA block and reaches all descendant sites
  // by SPIR-V dominance. The int heap path is intentionally NOT routed through this row claim: int adstacks back
  // loop-index recovery and if-branch flags that the autodiff pass emits unconditionally at the offload body root, and
  // `get_ad_stack_heap_thread_base_int()` keeps the eager `gl_GlobalInvocationID * stride_int` per-thread layout
  // instead of consulting any row_id_var.
  if (stmt == ad_stack_lca_block_float_ && ad_stack_lca_block_float_ != nullptr) {
    QD_ASSERT(ad_stack_row_id_var_float_.id == 0);
    ad_stack_row_id_var_float_ = ir_->alloca_variable(ir_->u32_type());
    ir_->store_variable(ad_stack_row_id_var_float_, ir_->uint_immediate_number(ir_->u32_type(), UINT32_MAX));
  }
  // Tasks without a captured `bound_expr` do not have a host-published row capacity and the float heap is sized at
  // `dispatched_threads * stride_float` worst case. Emitting the LCA-block atomic-rmw claim in that case lets
  // `claimed_row` exceed `dispatched_threads` whenever the kernel's iteration count exceeds the SPIR-V advisory cap
  // (`advisory_total_num_threads = 65536` for struct_for, `<= 131072` for range_for) and the kernel grid-strides via
  // `loop_var += total_invocs`, because every iteration that reaches the LCA increments the counter and the inert
  // UINT32_MAX-capacity clamp does not bring the row back in-bounds. Fall back to the eager `gl_GlobalInvocationID *
  // stride_float` mapping by storing the invocation id into `row_id_var_float` directly; downstream
  // `get_ad_stack_heap_thread_base_float()` reads it and produces the same per-thread addressing the int heap uses.
  if (stmt == ad_stack_lca_block_float_ && ad_stack_lca_block_float_ != nullptr &&
      !task_attribs_.ad_stack.bound_expr.has_value()) {
    spirv::Value invoc_id = ir_->get_global_invocation_id(0);
    ir_->store_variable(ad_stack_row_id_var_float_, invoc_id);
  } else if (stmt == ad_stack_lca_block_float_ && ad_stack_lca_block_float_ != nullptr) {
    if (ad_stack_row_counter_buffer_.id == 0) {
      ad_stack_row_counter_buffer_ = get_buffer_value({BufferType::AdStackRowCounter}, PrimitiveType::u32);
    }
    // Per-task slot: the host allocates the counter buffer as `uint[num_tasks_in_kernel]`, clears it once at the start
    // of each kernel-launch (not between tasks), so each task's atomic claims accumulate in its own slot and survive
    // until the post-launch host readback at `synchronize()`. Without per-task slots a single shared slot would have
    // the next task's bind-time clear destroy this task's count before the host can observe it, and the heap-sizing
    // path would only ever see the LAST task's claim count - useless for tasks that come earlier in a multi-task kernel
    // and have wildly different work patterns.
    spirv::Value counter_ptr = ir_->struct_array_access(
        ir_->u32_type(), ad_stack_row_counter_buffer_, ir_->uint_immediate_number(ir_->i32_type(), task_id_in_kernel_));
    spirv::Value claimed_row =
        ir_->make_value(spv::OpAtomicIAdd, ir_->u32_type(), counter_ptr,
                        /*scope=*/ir_->const_i32_one_,
                        /*semantics=*/ir_->const_i32_zero_, ir_->uint_immediate_number(ir_->u32_type(), 1));
    ir_->store_variable(ad_stack_row_id_var_float_, claimed_row);

    // Defense-in-depth bounds check. The host writes the per-task row capacity into
    // `BufferType::AdStackBoundRowCapacity[task_id]` before this dispatch starts: for tasks with a captured
    // `bound_expr`, the value is the exact reducer count; for every other task the value is
    // UINT32_MAX so this check is inert. When `claimed_row >= capacity` we OpAtomicUMax UINT32_MAX into the existing
    // AdStackOverflow buffer; the synchronize() readback recognises that sentinel and raises a clear actionable error
    // rather than letting the kernel silently OOB-write the heap. UINT32_MAX cannot collide with the existing per-stack
    // `stack_id+1` overflow signal because `stack_id+1 <= num_ad_stacks << UINT32_MAX` in every realistic kernel.
    // Expected behaviour on legitimate workloads: this branch is taken zero times. If it fires, the reducer's count
    // diverged from the main pass's actual LCA-block-reaching thread count, which means an internal-consistency bug
    // (non-determinism between reducer and main), not a user-recoverable condition. The clamp via OpSelect keeps the
    // stored row id in-bounds at `capacity-1` when the over-claim happens, so downstream push / load-top sites in this
    // overshooting thread do not write past the heap end.
    if (ad_stack_bound_row_capacity_buffer_.id == 0) {
      ad_stack_bound_row_capacity_buffer_ = get_buffer_value({BufferType::AdStackBoundRowCapacity}, PrimitiveType::u32);
    }
    spirv::Value capacity_ptr =
        ir_->struct_array_access(ir_->u32_type(), ad_stack_bound_row_capacity_buffer_,
                                 ir_->uint_immediate_number(ir_->i32_type(), task_id_in_kernel_));
    spirv::Value capacity = ir_->load_variable(capacity_ptr, ir_->u32_type());
    // Guard the `capacity - 1` clamp upper bound against `capacity == 0`: a naive `sub(capacity, 1)` wraps in u32 to
    // UINT32_MAX, the `UMin(claimed_row, UINT32_MAX)` returns `claimed_row` unchanged for any realistic value, and the
    // clamp goes inert. Clamp the upper bound to row 0 in that case (the launcher floors the heap allocation at one row
    // precisely so the single-slot fallback is always backed by real storage). Mirrors the LLVM-side `select(capacity
    // == 0, 0, capacity - 1)`.
    spirv::Value zero_u32 = ir_->uint_immediate_number(ir_->u32_type(), 0);
    spirv::Value one_u32 = ir_->uint_immediate_number(ir_->u32_type(), 1);
    spirv::Value capacity_is_zero = ir_->eq(capacity, zero_u32);
    spirv::Value capacity_minus_one_raw = ir_->sub(capacity, one_u32);
    spirv::Value clamp_upper = ir_->select(capacity_is_zero, zero_u32, capacity_minus_one_raw);
    spirv::Value clamped_row = ir_->call_glsl450(ir_->u32_type(), GLSLstd450UMin, claimed_row, clamp_upper);
    ir_->store_variable(ad_stack_row_id_var_float_, clamped_row);
    spirv::Value overflow_signal =
        ir_->select(ir_->ge(claimed_row, capacity), ir_->uint_immediate_number(ir_->u32_type(), UINT32_MAX),
                    ir_->uint_immediate_number(ir_->u32_type(), 0));
    spirv::Value overflow_buf = get_buffer_value(BufferType::AdStackOverflow, PrimitiveType::u32);
    spirv::Value overflow_ptr =
        ir_->struct_array_access(ir_->u32_type(), overflow_buf, ir_->uint_immediate_number(ir_->i32_type(), 0));
    ir_->make_value(spv::OpAtomicUMax, ir_->u32_type(), overflow_ptr, /*scope=*/ir_->const_i32_one_,
                    /*semantics=*/ir_->const_i32_zero_, overflow_signal);
  }
  for (auto &s : stmt->statements) {
    if (offload_loop_motion_.find(s.get()) == offload_loop_motion_.end()) {
      s->accept(this);
    }
  }
}

void TaskCodegen::visit(PrintStmt *stmt) {
  if (!caps_->get(DeviceCapability::spirv_has_non_semantic_info)) {
    return;
  }

  std::string formats;
  std::vector<Value> vals;

  for (auto i = 0; i < stmt->contents.size(); ++i) {
    auto const &content = stmt->contents[i];
    auto const &format = stmt->formats[i];
    if (std::holds_alternative<Stmt *>(content)) {
      auto arg_stmt = std::get<Stmt *>(content);
      QD_ASSERT(!arg_stmt->ret_type->is<TensorType>());

      auto value = ir_->query_value(arg_stmt->raw_name());
      vals.push_back(value);

      auto &&merged_format = merge_printf_specifier(format, data_type_format(arg_stmt->ret_type));
      // Vulkan doesn't support length, flags, or width specifier, except for
      // unsigned long.
      // https://vulkan.lunarg.com/doc/view/1.3.204.1/windows/debug_printf.html
      auto &&[format_flags, format_width, format_precision, format_length, format_conversion] =
          parse_printf_specifier(merged_format);
      if (!format_flags.empty()) {
        QD_WARN(
            "The printf flags '{}' are not supported in Vulkan, "
            "and will be discarded.",
            format_flags);
        format_flags.clear();
      }
      if (!format_width.empty()) {
        QD_WARN(
            "The printf width modifier '{}' is not supported in Vulkan, "
            "and will be discarded.",
            format_width);
        format_width.clear();
      }
      if (!format_length.empty() && !(format_length == "l" && (format_conversion == "u" || format_conversion == "x"))) {
        QD_WARN(
            "The printf length modifier '{}' is not supported in Vulkan, "
            "and will be discarded.",
            format_length);
        format_length.clear();
      }
      formats += "%" + format_precision.append(format_length).append(format_conversion);
    } else {
      auto arg_str = std::get<std::string>(content);
      formats += sanitize_format_string(arg_str);
    }
  }
  ir_->call_debugprintf(formats, vals);
}

void TaskCodegen::visit(ConstStmt *const_stmt) {
  auto get_const = [&](const TypedConstant &const_val) {
    auto dt = const_val.dt.ptr_removed();
    spirv::SType stype = ir_->get_primitive_type(dt);

    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      return ir_->float_immediate_number(stype, static_cast<double>(const_val.val_f32), false);
    } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
      // Ref: See quadrants::lang::TypedConstant::TypedConstant()
      // FP16 is stored as FP32 on host side,
      // as some CPUs does not have native FP16 (and no libc support)
      return ir_->float_immediate_number(stype, static_cast<double>(const_val.val_f32), false);
    } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
      return ir_->int_immediate_number(stype, static_cast<int64_t>(const_val.val_i32), false);
    } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
      return ir_->int_immediate_number(stype, static_cast<int64_t>(const_val.val_i64), false);
    } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
      return ir_->float_immediate_number(stype, static_cast<double>(const_val.val_f64), false);
    } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
      return ir_->int_immediate_number(stype, static_cast<int64_t>(const_val.val_i8), false);
    } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
      return ir_->int_immediate_number(stype, static_cast<int64_t>(const_val.val_i16), false);
    } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
      return ir_->uint_immediate_number(stype, static_cast<uint64_t>(const_val.val_u1), false);
    } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
      return ir_->uint_immediate_number(stype, static_cast<uint64_t>(const_val.val_u8), false);
    } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
      return ir_->uint_immediate_number(stype, static_cast<uint64_t>(const_val.val_u16), false);
    } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
      return ir_->uint_immediate_number(stype, static_cast<uint64_t>(const_val.val_u32), false);
    } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
      return ir_->uint_immediate_number(stype, static_cast<uint64_t>(const_val.val_u64), false);
    } else {
      QD_P(data_type_name(dt));
      QD_NOT_IMPLEMENTED
      return spirv::Value();
    }
  };

  spirv::Value val = get_const(const_stmt->val);
  ir_->register_value(const_stmt->raw_name(), val);
}

void TaskCodegen::visit(AllocaStmt *alloca) {
  spirv::Value ptr_val;
  // alloca->ret_type is a pointer to the stored type; ptr_removed() gives the
  // stored type itself (e.g. TensorType<32 x f32> for a 32-element array).
  auto alloca_type = alloca->ret_type.ptr_removed();
  // Shared array is always modeled as a tensor type, i.e. an array of scalars.
  if (auto tensor_type = alloca_type->cast<TensorType>()) {
    // Do NOT initialize elem_num/elem_type here - the helper flattens nested
    // tensor types (e.g. vec3 -> 3xf32) before computing them. Pre-initializing
    // with get_primitive_type(tensor_type->get_element_type()) would crash on
    // nested tensor types like Tensor(3) f32.
    int elem_num;
    spirv::SType elem_type;
    maybe_retype_alloca(*ir_, *caps_, alloca, tensor_type, shared_float_allocas_with_atomic_rmw_,
                        uint_backed_shared_float_ptr_stmts_, elem_num, elem_type);
    // Use `get_function_array_type` rather than `get_array_type`: the resulting array type backs an
    // `OpVariable` in either `Workgroup` (shared) or `Function` storage class, neither of which is a
    // storage-buffer / PSB / Uniform interface, so the `ArrayStride` decoration `get_array_type` adds
    // would be illegal per `VUID-StandaloneSpirv-None-10684` and Blackwell-class NVIDIA Vulkan drivers
    // refuse the resulting compute pipeline. Older drivers tolerated the over-decoration.
    spirv::SType arr_type = ir_->get_function_array_type(elem_type, elem_num);
    if (alloca->is_shared) {  // for shared memory / workgroup memory
      ptr_val = ir_->alloca_workgroup_array(arr_type);
      shared_array_binds_.push_back(ptr_val);
    } else {  // for function memory
      ptr_val = ir_->alloca_variable(arr_type);
    }
  } else {
    // Alloca for a single variable
    spirv::SType src_type = ir_->get_primitive_type(alloca_type);
    ptr_val = ir_->alloca_variable(src_type);
    ir_->store_variable(ptr_val, ir_->get_zero(src_type));
  }
  ir_->register_value(alloca->raw_name(), ptr_val);
}

void TaskCodegen::visit(MatrixPtrStmt *stmt) {
  spirv::Value ptr_val;
  spirv::Value origin_val = ir_->query_value(stmt->origin->raw_name());
  spirv::Value offset_val = ir_->query_value(stmt->offset->raw_name());
  auto dt = stmt->element_type().ptr_removed();
  if (stmt->offset_used_as_index()) {
    // Origin is a local/shared array allocation or a derived pointer from one
    // - use OpAccessChain or OpPtrAccessChain respectively.
    if (stmt->origin->is<AllocaStmt>() || origin_val.stype.flag == TypeKind::kPtr) {
      maybe_retype_derived_ptr(*ir_, stmt->origin, stmt, dt, uint_backed_shared_float_ptr_stmts_);
      spirv::SType ptr_type = ir_->get_pointer_type(ir_->get_primitive_type(dt), origin_val.stype.storage_class);
      auto op = stmt->origin->is<AllocaStmt>() ? spv::OpAccessChain : spv::OpPtrAccessChain;
      ptr_val = ir_->make_value(op, ptr_type, origin_val, offset_val);
      if (auto *a = stmt->origin->cast<AllocaStmt>(); a && a->is_shared) {
        ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
      }
    } else if (stmt->origin->is<GlobalTemporaryStmt>()) {
      spirv::Value dt_bytes = ir_->int_immediate_number(ir_->i32_type(), ir_->get_primitive_type_size(dt), false);
      spirv::Value offset_bytes = ir_->mul(dt_bytes, offset_val);
      ptr_val = ir_->add(origin_val, offset_bytes);
      ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
    } else {
      QD_NOT_IMPLEMENTED;
    }
  } else {  // offset used as bytes
    ptr_val = ir_->add(origin_val, ir_->cast(origin_val.stype, offset_val));
    ptr_to_buffers_[stmt] = ptr_to_buffers_[stmt->origin];
  }
  ir_->register_value(stmt->raw_name(), ptr_val);
}

void TaskCodegen::visit(LocalLoadStmt *stmt) {
  auto ptr = stmt->src;
  spirv::Value ptr_val = ir_->query_value(ptr->raw_name());
  spirv::Value val;
  if (uint_backed_shared_float_ptr_stmts_.count(ptr)) {
    val = load_uint_backed_shared_float(*ir_, ptr_val, stmt->element_type());
  } else {
    val = ir_->load_variable(ptr_val, ir_->get_primitive_type(stmt->element_type()));
  }
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(LocalStoreStmt *stmt) {
  spirv::Value ptr_val = ir_->query_value(stmt->dest->raw_name());
  spirv::Value val = ir_->query_value(stmt->val->raw_name());
  if (uint_backed_shared_float_ptr_stmts_.count(stmt->dest)) {
    val = float_to_shared_uint(*ir_, val, stmt->val->element_type());
  }
  ir_->store_variable(ptr_val, val);
}

void TaskCodegen::visit(GetRootStmt *stmt) {
  const int root_id = snode_to_root_.at(stmt->root()->id);
  root_stmts_[root_id] = stmt;
  // get_buffer_value({BufferType::Root, root_id}, PrimitiveType::u32);
  spirv::Value root_val = make_pointer(0);
  ir_->register_value(stmt->raw_name(), root_val);
}

void TaskCodegen::visit(GetChStmt *stmt) {
  // TODO: GetChStmt -> GetComponentStmt ?
  const int root = snode_to_root_.at(stmt->input_snode->id);

  const auto &snode_descs = compiled_structs_[root].snode_descriptors;
  auto *out_snode = stmt->output_snode;
  QD_ASSERT(snode_descs.at(stmt->input_snode->id).get_child(stmt->chid) == out_snode);

  const auto &desc = snode_descs.at(out_snode->id);

  spirv::Value input_ptr_val = ir_->query_value(stmt->input_ptr->raw_name());
  spirv::Value offset = make_pointer(desc.mem_offset_in_parent_cell);
  spirv::Value val = ir_->add(input_ptr_val, offset);
  ir_->register_value(stmt->raw_name(), val);

  if (out_snode->is_place()) {
    QD_ASSERT(ptr_to_buffers_.count(stmt) == 0);
    ptr_to_buffers_[stmt] = BufferInfo(BufferType::Root, root);
  }
}

enum class ActivationOp { activate, deactivate, query };

spirv::Value TaskCodegen::bitmasked_activation(ActivationOp op,
                                               spirv::Value parent_ptr,
                                               int root_id,
                                               const SNode *sn,
                                               spirv::Value input_index) {
  spirv::SType ptr_dt = parent_ptr.stype;
  const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
  const auto &desc = snode_descs.at(sn->id);

  auto bitmask_word_index =
      ir_->make_value(spv::OpShiftRightLogical, ptr_dt, input_index, ir_->uint_immediate_number(ptr_dt, 5));
  auto bitmask_bit_index =
      ir_->make_value(spv::OpBitwiseAnd, ptr_dt, input_index, ir_->uint_immediate_number(ptr_dt, 31));
  auto bitmask_mask = ir_->make_value(spv::OpShiftLeftLogical, ptr_dt, ir_->const_i32_one_, bitmask_bit_index);

  auto buffer = get_buffer_value(BufferInfo(BufferType::Root, root_id), PrimitiveType::u32);
  auto bitmask_word_ptr = ir_->make_value(spv::OpShiftLeftLogical, ptr_dt, bitmask_word_index,
                                          ir_->uint_immediate_number(ir_->u32_type(), 2));
  bitmask_word_ptr = ir_->add(bitmask_word_ptr, make_pointer(desc.cell_stride * desc.snode->num_cells_per_container));
  bitmask_word_ptr = ir_->add(parent_ptr, bitmask_word_ptr);
  bitmask_word_ptr = ir_->make_value(spv::OpShiftRightLogical, ir_->u32_type(), bitmask_word_ptr,
                                     ir_->uint_immediate_number(ir_->u32_type(), 2));
  bitmask_word_ptr = ir_->struct_array_access(ir_->u32_type(), buffer, bitmask_word_ptr);

  if (op == ActivationOp::activate) {
    return ir_->make_value(spv::OpAtomicOr, ir_->u32_type(), bitmask_word_ptr,
                           /*scope=*/ir_->const_i32_one_,
                           /*semantics=*/ir_->const_i32_zero_, bitmask_mask);
  } else if (op == ActivationOp::deactivate) {
    bitmask_mask = ir_->make_value(spv::OpNot, ir_->u32_type(), bitmask_mask);
    return ir_->make_value(spv::OpAtomicAnd, ir_->u32_type(), bitmask_word_ptr,
                           /*scope=*/ir_->const_i32_one_,
                           /*semantics=*/ir_->const_i32_zero_, bitmask_mask);
  } else {
    auto bitmask_val = ir_->load_variable(bitmask_word_ptr, ir_->u32_type());
    auto bit = ir_->make_value(spv::OpShiftRightLogical, ir_->u32_type(), bitmask_val, bitmask_bit_index);
    bit = ir_->make_value(spv::OpBitwiseAnd, ir_->u32_type(), bit, ir_->uint_immediate_number(ir_->u32_type(), 1));
    return ir_->make_value(spv::OpUGreaterThan, ir_->bool_type(), bit, ir_->uint_immediate_number(ir_->u32_type(), 0));
  }
}

void TaskCodegen::visit(SNodeOpStmt *stmt) {
  const int root_id = snode_to_root_.at(stmt->snode->id);
  std::string parent = stmt->ptr->raw_name();
  spirv::Value parent_val = ir_->query_value(parent);

  if (stmt->snode->type == SNodeType::bitmasked) {
    spirv::Value input_index_val = ir_->cast(parent_val.stype, ir_->query_value(stmt->val->raw_name()));

    if (stmt->op_type == SNodeOpType::is_active) {
      auto is_active = bitmasked_activation(ActivationOp::query, parent_val, root_id, stmt->snode, input_index_val);
      is_active = ir_->cast(ir_->get_primitive_type(stmt->ret_type), is_active);
      is_active = ir_->make_value(spv::OpSNegate, is_active.stype, is_active);
      ir_->register_value(stmt->raw_name(), is_active);
    } else if (stmt->op_type == SNodeOpType::deactivate) {
      bitmasked_activation(ActivationOp::deactivate, parent_val, root_id, stmt->snode, input_index_val);
    } else if (stmt->op_type == SNodeOpType::activate) {
      bitmasked_activation(ActivationOp::activate, parent_val, root_id, stmt->snode, input_index_val);
    } else {
      QD_NOT_IMPLEMENTED;
    }
  } else {
    QD_NOT_IMPLEMENTED;
  }
}

void TaskCodegen::visit(SNodeLookupStmt *stmt) {
  // TODO: SNodeLookupStmt -> GetSNodeCellStmt ?
  bool is_root{false};  // Eliminate first root snode access
  const int root_id = snode_to_root_.at(stmt->snode->id);
  std::string parent;

  if (stmt->input_snode) {
    parent = stmt->input_snode->raw_name();
  } else {
    QD_ASSERT(root_stmts_.at(root_id) != nullptr);
    parent = root_stmts_.at(root_id)->raw_name();
  }
  const auto *sn = stmt->snode;

  spirv::Value parent_val = ir_->query_value(parent);

  if (stmt->activate) {
    if (sn->type == SNodeType::dense) {
      // Do nothing
    } else if (sn->type == SNodeType::bitmasked) {
      spirv::Value input_index_val = ir_->query_value(stmt->input_index->raw_name());
      bitmasked_activation(ActivationOp::activate, parent_val, root_id, sn, input_index_val);
    } else {
      QD_NOT_IMPLEMENTED;
    }
  }

  spirv::Value val;
  if (is_root) {
    val = parent_val;  // Assert Root[0] access at first time
  } else {
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);

    spirv::Value input_index_val = ir_->cast(parent_val.stype, ir_->query_value(stmt->input_index->raw_name()));
    spirv::Value stride = make_pointer(desc.cell_stride);
    spirv::Value offset = ir_->mul(input_index_val, stride);
    val = ir_->add(parent_val, offset);
  }
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(RandStmt *stmt) {
  spirv::Value val;
  spirv::Value global_tmp = get_buffer_value(BufferType::GlobalTmps, PrimitiveType::u32);
  if (stmt->element_type()->is_primitive(PrimitiveTypeID::i32)) {
    val = ir_->rand_i32(global_tmp);
  } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::u32)) {
    val = ir_->rand_u32(global_tmp);
  } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::f32)) {
    val = ir_->rand_f32(global_tmp);
  } else if (stmt->element_type()->is_primitive(PrimitiveTypeID::f16)) {
    auto highp_val = ir_->rand_f32(global_tmp);
    val = ir_->cast(ir_->f16_type(), highp_val);
  } else {
    QD_ERROR("rand only support 32-bit type");
  }
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(LinearizeStmt *stmt) {
  spirv::Value val = ir_->const_i32_zero_;
  for (size_t i = 0; i < stmt->inputs.size(); ++i) {
    spirv::Value strides_val = ir_->int_immediate_number(ir_->i32_type(), stmt->strides[i]);
    spirv::Value input_val = ir_->query_value(stmt->inputs[i]->raw_name());
    val = ir_->add(ir_->mul(val, strides_val), input_val);
  }
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(LoopIndexStmt *stmt) {
  const auto stmt_name = stmt->raw_name();
  if (stmt->loop->is<OffloadedStmt>()) {
    const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
    if (type == OffloadedTaskType::range_for) {
      QD_ASSERT(stmt->index == 0);
      spirv::Value loop_var = ir_->query_value("ii");
      // spirv::Value val = ir_->add(loop_var, ir_->const_i32_zero_);
      ir_->register_value(stmt_name, loop_var);
    } else {
      QD_NOT_IMPLEMENTED;
    }
  } else if (stmt->loop->is<RangeForStmt>()) {
    QD_ASSERT(stmt->index == 0);
    spirv::Value loop_var = ir_->query_value(stmt->loop->raw_name());
    spirv::Value val = ir_->add(loop_var, ir_->const_i32_zero_);
    ir_->register_value(stmt_name, val);
  } else {
    QD_NOT_IMPLEMENTED;
  }
}

void TaskCodegen::visit(GlobalStoreStmt *stmt) {
  spirv::Value val = ir_->query_value(stmt->val->raw_name());

  store_buffer(stmt->dest, val);
}

void TaskCodegen::visit(GlobalLoadStmt *stmt) {
  auto dt = stmt->element_type();

  auto val = load_buffer(stmt->src, dt, stmt->is_volatile);

  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(ArgLoadStmt *stmt) {
  const auto arg_id = stmt->arg_id;
  const std::vector<int> indices_l(stmt->arg_id.begin(), stmt->arg_id.begin());
  const std::vector<int> indices_r(stmt->arg_id.begin(), stmt->arg_id.end());
  const auto arg_type = ctx_attribs_->args_type()->get_element_type(arg_id);
  if (arg_type->is<PointerType>() ||
      (arg_type->is<lang::StructType>() && arg_type->as<lang::StructType>()->elements().size() >= 2 &&
       arg_type->as<lang::StructType>()->get_element_type(std::array<int, 1>{1})->is<PointerType>())) {
    // Do not shift! We are indexing the buffers at byte granularity.
    // spirv::Value val =
    //    ir_->int_immediate_number(ir_->i32_type(), offset_in_mem);
    // ir_->register_value(stmt->raw_name(), val);
  } else {
    spirv::Value buffer_val, buffer_value;
    bool is_bool = arg_type->is_primitive(PrimitiveTypeID::u1);
    // `val_type` must be assigned after `get_buffer_value` because
    // `args_struct_types_` needs to be initialized by `get_buffer_value`.
    SType val_type;

    buffer_value = get_buffer_value(BufferType::Args, PrimitiveType::i32);
    val_type = is_bool ? ir_->i32_type() : args_struct_types_[arg_id];
    buffer_val =
        ir_->make_access_chain(ir_->get_pointer_type(val_type, spv::StorageClassUniform), buffer_value, arg_id);
    buffer_val.flag = ValueKind::kVariablePtr;
    if (!stmt->create_load) {
      ir_->register_value(stmt->raw_name(), buffer_val);
      return;
    }
    spirv::Value val = ir_->load_variable(buffer_val, val_type);
    if (is_bool) {
      val = ir_->make_value(spv::OpINotEqual, ir_->bool_type(), val, ir_->int_immediate_number(ir_->i32_type(), 0));
    }
    ir_->register_value(stmt->raw_name(), val);
  }
}

void TaskCodegen::visit(GetElementStmt *stmt) {
  spirv::Value val = ir_->query_value(stmt->src->raw_name());
  const auto val_type = ir_->get_primitive_type(stmt->element_type());
  const auto val_type_ptr = ir_->get_pointer_type(val_type, spv::StorageClassUniform);
  val = ir_->make_access_chain(val_type_ptr, val, stmt->index);
  val = ir_->load_variable(val, val_type);
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(ReturnStmt *stmt) {
  QD_ASSERT(ctx_attribs_->has_rets());
  // The `PrimitiveType::i32` in this function call is a placeholder.
  auto buffer_value = get_buffer_value(BufferType::Rets, PrimitiveType::i32);
  // Function to store variable using indices provided by
  // `calc_indices_and_store`.
  auto store_variable = [&](int index, const std::vector<int> &indices) {
    auto dt = stmt->element_types()[index];
    auto val_type = ir_->get_primitive_type(dt);
    // Extend u1 values to i32 to be passed to the host.
    if (dt->is_primitive(PrimitiveTypeID::u1))
      val_type = ir_->i32_type();
    spirv::Value buffer_val;
    // Accessing based on `indices` using OpAccessChain.
    buffer_val = ir_->make_access_chain(ir_->get_storage_pointer_type(val_type), buffer_value, indices);
    buffer_val.flag = ValueKind::kVariablePtr;
    spirv::Value val = ir_->query_value(stmt->values[index]->raw_name());
    // Extend u1 values to i32 to be passed to the host.
    if (dt->is_primitive(PrimitiveTypeID::u1))
      val = ir_->select(val, ir_->const_i32_one_, ir_->const_i32_zero_);
    ir_->store_variable(buffer_val, val);
  };
  // Function to traverse struct tree in depth-first order recursively to
  // calculate AccessChain indices.
  std::function<void(const quadrants::lang::Type *, int &, std::vector<int> &)> calc_indices_and_store =
      [&](const quadrants::lang::Type *type, int &index, std::vector<int> &indices) {
        if (auto struct_type = type->cast<quadrants::lang::StructType>()) {
          for (int i = 0; i < struct_type->elements().size(); ++i) {
            indices.push_back(i);
            calc_indices_and_store(struct_type->elements()[i].type, index, indices);
            indices.pop_back();
          }
        } else if (auto tensor_type = type->cast<quadrants::lang::TensorType>()) {
          int num = tensor_type->get_num_elements();
          for (int i = 0; i < num; ++i) {
            indices.push_back(i);
            store_variable(index++, indices);
            indices.pop_back();
          }
        } else {
          store_variable(index++, indices);
        }
      };
  // Launch depth-first traversal using `calc_indices_and_store` on return
  // struct.
  std::vector<int> indices;
  int index = 0;
  for (int i = 0; i < ctx_attribs_->rets_type()->elements().size(); ++i) {
    indices.push_back(i);
    calc_indices_and_store(ctx_attribs_->rets_type()->elements()[i].type, index, indices);
    indices.pop_back();
  }
}

void TaskCodegen::visit(GlobalTemporaryStmt *stmt) {
  spirv::Value val = ir_->int_immediate_number(ir_->i32_type(), stmt->offset,
                                               false);  // Named Constant
  ir_->register_value(stmt->raw_name(), val);
  ptr_to_buffers_[stmt] = BufferType::GlobalTmps;
}

void TaskCodegen::visit(ExternalTensorShapeAlongAxisStmt *stmt) {
  const auto name = stmt->raw_name();
  const auto arg_id = stmt->arg_id;
  const auto axis = stmt->axis;

  spirv::Value var_ptr;
  QD_ASSERT(ctx_attribs_->args_type()->get_element_type(arg_id)->is<lang::StructType>());
  std::vector<int> indices = arg_id;
  indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
  indices.push_back(axis);
  var_ptr = ir_->make_access_chain(ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
                                   get_buffer_value(BufferType::Args, PrimitiveType::i32), indices);
  spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());

  ir_->register_value(name, var);
}

void TaskCodegen::visit(ExternalPtrStmt *stmt) {
  // Used mostly for transferring data between host (e.g. numpy array) and
  // device.
  spirv::Value linear_offset = ir_->int_immediate_number(ir_->i32_type(), 0);
  const auto *argload = stmt->base_ptr->as<ArgLoadStmt>();
  const auto arg_id = argload->arg_id;
  {
    const int num_indices = stmt->indices.size();
    std::vector<std::string> size_var_names;
    const auto &element_shape = stmt->element_shape;
    const size_t element_shape_index_offset = num_indices - element_shape.size();
    for (int i = 0; i < num_indices - element_shape.size(); i++) {
      std::string var_name = fmt::format("{}_size{}_", stmt->raw_name(), i);
      std::vector<int> indices = arg_id;
      indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      indices.push_back(i);
      spirv::Value var_ptr = ir_->make_access_chain(ir_->get_pointer_type(ir_->i32_type(), spv::StorageClassUniform),
                                                    get_buffer_value(BufferType::Args, PrimitiveType::i32), indices);
      spirv::Value var = ir_->load_variable(var_ptr, ir_->i32_type());
      ir_->register_value(var_name, var);
      size_var_names.push_back(std::move(var_name));
    }
    int size_var_names_idx = 0;
    for (int i = 0; i < num_indices; i++) {
      spirv::Value size_var;
      // Use immediate numbers to flatten index for element shapes.
      if (i >= element_shape_index_offset && i < element_shape_index_offset + element_shape.size()) {
        size_var = ir_->uint_immediate_number(ir_->i32_type(), element_shape[i - element_shape_index_offset]);
      } else {
        size_var = ir_->query_value(size_var_names[size_var_names_idx++]);
      }
      spirv::Value indices = ir_->query_value(stmt->indices[i]->raw_name());
      linear_offset = ir_->mul(linear_offset, size_var);
      linear_offset = ir_->add(linear_offset, indices);
    }
    size_t type_size = ir_->get_primitive_type_size(stmt->ret_type.ptr_removed());
    linear_offset = ir_->make_value(spv::OpShiftLeftLogical, ir_->i32_type(), linear_offset,
                                    ir_->int_immediate_number(ir_->i32_type(), log2int(type_size)));
    if (caps_->get(DeviceCapability::spirv_has_no_integer_wrap_decoration)) {
      ir_->decorate(spv::OpDecorate, linear_offset, spv::DecorationNoSignedWrap);
    }
  }
  if (caps_->get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    std::vector<int> indices = arg_id;
    // Pick the data or gradient pointer slot of the ndarray argument struct. Without this, reverse-mode AD kernels
    // accumulate into x.data instead of x.grad and host-side gradients stay at zero.
    indices.push_back(stmt->is_grad ? TypeFactory::GRAD_PTR_POS_IN_NDARRAY : TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    spirv::Value addr_ptr = ir_->make_access_chain(ir_->get_pointer_type(ir_->u64_type(), spv::StorageClassUniform),
                                                   get_buffer_value(BufferType::Args, PrimitiveType::i32), indices);
    spirv::Value base_addr = ir_->load_variable(addr_ptr, ir_->u64_type());
    spirv::Value addr = ir_->add(base_addr, ir_->make_value(spv::OpSConvert, ir_->u64_type(), linear_offset));
    ir_->register_value(stmt->raw_name(), addr);

    // Save decomposed base pointer and element index so at_buffer() can
    // emit OpConvertUToPtr on the base address once and use OpPtrAccessChain
    // for element access.  This avoids a Metal shader compiler bug where
    // per-element reinterpret_cast from ulong arithmetic is miscompiled
    // when the stored value is loop-invariant.
    size_t type_size = ir_->get_primitive_type_size(stmt->ret_type.ptr_removed());
    spirv::Value elem_index = ir_->make_value(spv::OpShiftRightLogical, ir_->i32_type(), linear_offset,
                                              ir_->int_immediate_number(ir_->i32_type(), log2int(type_size)));
    physical_ptr_components_[stmt] = {base_addr, elem_index};
  } else {
    ir_->register_value(stmt->raw_name(), linear_offset);
  }

  if (ctx_attribs_->arg_at(arg_id).is_array) {
    QD_ASSERT(arg_id.size() == 1);
    ptr_to_buffers_[stmt] = {BufferType::ExtArr, arg_id[0], stmt->is_grad};
  } else {
    ptr_to_buffers_[stmt] = BufferType::Args;
  }
}

void TaskCodegen::visit(DecorationStmt *stmt) {
}

void TaskCodegen::visit(UnaryOpStmt *stmt) {
  const auto operand_name = stmt->operand->raw_name();

  const auto src_dt = stmt->operand->element_type();
  const auto dst_dt = stmt->element_type();
  spirv::SType src_type = ir_->get_primitive_type(src_dt);
  spirv::SType dst_type;
  if (dst_dt.is_pointer()) {
    auto stype = dst_dt.ptr_removed()->as<lang::StructType>();
    std::vector<std::tuple<SType, std::string, size_t>> components;
    for (int i = 0; i < stype->elements().size(); i++) {
      components.push_back({ir_->get_primitive_type(stype->get_element_type(std::array{i})),
                            fmt::format("element{}", i), stype->get_element_offset(std::array{i})});
    }
    dst_type = ir_->create_struct_type(components);
  } else {
    dst_type = ir_->get_primitive_type(dst_dt);
  }
  spirv::Value operand_val = ir_->query_value(operand_name);
  spirv::Value val = spirv::Value();

  if (stmt->op_type == UnaryOpType::logic_not) {
    spirv::Value zero = ir_->get_zero(src_type);  // Math zero type to left hand side
    if (is_integral(src_dt)) {
      if (is_signed(src_dt)) {
        zero = ir_->int_immediate_number(src_type, 0);
      } else {
        zero = ir_->uint_immediate_number(src_type, 0);
      }
    } else if (is_real(src_dt)) {
      zero = ir_->float_immediate_number(src_type, 0);
    } else {
      QD_NOT_IMPLEMENTED
    }
    val = ir_->cast(dst_type, ir_->eq(operand_val, zero));
  } else if (stmt->op_type == UnaryOpType::neg) {
    operand_val = ir_->cast(dst_type, operand_val);
    if (is_integral(dst_dt)) {
      if (is_signed(dst_dt)) {
        val = ir_->make_value(spv::OpSNegate, dst_type, operand_val);
      } else {
        QD_NOT_IMPLEMENTED
      }
    } else if (is_real(dst_dt)) {
      val = ir_->make_value(spv::OpFNegate, dst_type, operand_val);
    } else {
      QD_NOT_IMPLEMENTED
    }
  } else if (stmt->op_type == UnaryOpType::rsqrt) {
    const uint32_t InverseSqrt_id = 32;
    if (is_real(src_dt)) {
      val = ir_->call_glsl450(src_type, InverseSqrt_id, operand_val);
      val = ir_->cast(dst_type, val);
    } else {
      QD_NOT_IMPLEMENTED
    }
  } else if (stmt->op_type == UnaryOpType::sgn) {
    const uint32_t FSign_id = 6;
    const uint32_t SSign_id = 7;
    if (is_integral(src_dt)) {
      if (is_signed(src_dt)) {
        val = ir_->call_glsl450(src_type, SSign_id, operand_val);
      } else {
        QD_NOT_IMPLEMENTED
      }
    } else if (is_real(src_dt)) {
      val = ir_->call_glsl450(src_type, FSign_id, operand_val);
    } else {
      QD_NOT_IMPLEMENTED
    }
    val = ir_->cast(dst_type, val);
  } else if (stmt->op_type == UnaryOpType::bit_not) {
    operand_val = ir_->cast(dst_type, operand_val);
    if (is_integral(dst_dt)) {
      val = ir_->make_value(spv::OpNot, dst_type, operand_val);
    } else {
      QD_NOT_IMPLEMENTED
    }
  } else if (stmt->op_type == UnaryOpType::cast_value) {
    val = ir_->cast(dst_type, operand_val);
  } else if (stmt->op_type == UnaryOpType::cast_bits) {
    if (data_type_bits(src_dt) == data_type_bits(dst_dt)) {
      val = ir_->make_value(spv::OpBitcast, dst_type, operand_val);
    } else {
      QD_ERROR("bit_cast is only supported between data type with same size");
    }
  } else if (stmt->op_type == UnaryOpType::abs) {
    const uint32_t FAbs_id = 4;
    const uint32_t SAbs_id = 5;
    if (src_type.id == dst_type.id) {
      if (is_integral(src_dt)) {
        if (is_signed(src_dt)) {
          val = ir_->call_glsl450(src_type, SAbs_id, operand_val);
        } else {
          QD_NOT_IMPLEMENTED
        }
      } else if (is_real(src_dt)) {
        val = ir_->call_glsl450(src_type, FAbs_id, operand_val);
      } else {
        QD_NOT_IMPLEMENTED
      }
    } else {
      QD_NOT_IMPLEMENTED
    }
  } else if (stmt->op_type == UnaryOpType::inv) {
    if (is_real(dst_dt)) {
      // Do not pass `stmt->precise` to the builder here: the post-hoc `maybe_no_contraction(val, stmt->precise)`
      // block at the end of this visit() is the single source of truth for decoration, so passing `precise` at
      // creation time would emit a duplicate OpDecorate on the same OpFDiv value ID.
      val = ir_->div(ir_->float_immediate_number(dst_type, 1), operand_val);
    } else {
      QD_NOT_IMPLEMENTED
    }
  } else if (stmt->op_type == UnaryOpType::frexp) {
    // FrexpStruct is the same type of the first member.
    val = ir_->alloca_variable(dst_type);
    auto v = ir_->call_glsl450(dst_type, 52, operand_val);
    ir_->store_variable(val, v);
  } else if (stmt->op_type == UnaryOpType::popcnt) {
    // OpBitCount returns the operand's type, so for a u64 input it produces a u64 result. type_check normalises
    // the stmt's ret_type to i32 across every backend (CUDA / AMDGPU already do this in hardware), so we cast
    // the OpBitCount result down to dst_type (== i32) here. For an i32 / u32 input the cast is a free OpBitcast.
    val = ir_->cast(dst_type, ir_->popcnt(operand_val));
  } else if (stmt->op_type == UnaryOpType::clz) {
    // Use FindUMsb (75) rather than FindSMsb (74): clz() must count leading zeros over the unsigned bit pattern,
    // i.e. clz(0xFFFFFFFF) == 0. FindSMsb returns -1 for negative inputs (it finds the MSB of the absolute value's
    // bit pattern, ignoring the sign bit), which would yield clz(-1) == 32. CUDA's __nv_clz and the LLVM ctlz
    // intrinsic both operate on the unsigned bit pattern; FindUMsb gives matching semantics.
    //
    // All arithmetic happens in i32 regardless of the operand's signedness or width, then the result is cast
    // to dst_type at the very end. This keeps i32 / u32 / i64 / u64 inputs on the same code path: dispatch on
    // bit width only.
    uint32_t FindUMsb_id = 75;
    auto i32_t = ir_->i32_type();
    spirv::Value clz_i32;
    if (data_type_bits(src_dt) == 64) {
      // GLSL.std.450 FindUMsb is defined for 32-bit integers only. Synthesise the 64-bit case
      // by splitting the operand into hi/lo i32 halves, calling FindUMsb on each, and selecting:
      //   if hi != 0:  clz = 31 - FindUMsb(hi)        in [0, 31]
      //   else:        clz = 32 + (31 - FindUMsb(lo)) in [32, 64]
      // FindUMsb returns -1 on a zero input, so the "all-zero" cases fall out naturally:
      //   hi == 0, lo == 0 -> 32 + (31 - (-1)) = 64
      //   hi == 0, lo != 0 -> 32 + (31 - FindUMsb(lo))
      //   hi != 0          -> 31 - FindUMsb(hi)
      auto u64_t = ir_->u64_type();
      auto val_u64 = ir_->cast(u64_t, operand_val);
      auto thirty_two_u64 = ir_->uint_immediate_number(u64_t, 32);
      auto hi_u64 = ir_->make_value(spv::OpShiftRightLogical, u64_t, val_u64, thirty_two_u64);
      auto hi = ir_->cast(i32_t, hi_u64);
      auto lo = ir_->cast(i32_t, val_u64);
      auto hi_msb = ir_->call_glsl450(i32_t, FindUMsb_id, hi);
      auto lo_msb = ir_->call_glsl450(i32_t, FindUMsb_id, lo);
      auto bit31 = ir_->int_immediate_number(i32_t, 31);
      auto bit63 = ir_->int_immediate_number(i32_t, 63);
      auto zero_i32 = ir_->int_immediate_number(i32_t, 0);
      auto hi_clz = ir_->sub(bit31, hi_msb);
      auto lo_clz_full = ir_->sub(bit63, lo_msb);
      auto hi_zero = ir_->eq(hi, zero_i32);
      clz_i32 = ir_->select(hi_zero, lo_clz_full, hi_clz);
    } else if (data_type_bits(src_dt) == 32) {
      // Cast operand to i32 so FindUMsb's result type matches our i32 arithmetic. For i32 input this is a
      // no-op; for u32 input cast() emits an OpBitcast.
      auto val_i32 = ir_->cast(i32_t, operand_val);
      auto msb = ir_->call_glsl450(i32_t, FindUMsb_id, val_i32);
      auto bit31 = ir_->int_immediate_number(i32_t, 31);
      clz_i32 = ir_->sub(bit31, msb);
    } else {
      QD_NOT_IMPLEMENTED
    }
    // dst_type is i32 across every backend (set by type_check for popcnt / clz / ffs), so this cast is a
    // no-op for the i32 result we just computed; ir_->cast() returns the value unchanged when types match.
    val = ir_->cast(dst_type, clz_i32);
  } else if (stmt->op_type == UnaryOpType::ffs) {
    // ffs(x): 1-indexed position of the lowest set bit in x; 0 when x == 0 (CUDA __ffs convention).
    // GLSL.std.450 FindILsb (id 73) returns the 0-indexed lowest set bit, or -1 on a zero input. We map:
    //   ffs(x) = (x == 0) ? 0 : FindILsb(x) + 1
    // All arithmetic in i32, then cast back to dst_type. 64-bit inputs are synthesised by inspecting the
    // low half first (since "first" = lowest-indexed bit); if the low half is zero we use the high half
    // offset by 32. The bias has to be 32 (not 33) when applied to FindILsb(hi) directly, since +1 is
    // built into the lo-half arm via `lo_lsb + 1`; for the hi-half arm we use `hi_lsb + 33`.
    uint32_t FindILsb_id = 73;
    auto i32_t = ir_->i32_type();
    auto zero_i32 = ir_->int_immediate_number(i32_t, 0);
    auto one_i32 = ir_->int_immediate_number(i32_t, 1);
    spirv::Value ffs_i32;
    if (data_type_bits(src_dt) == 64) {
      auto u64_t = ir_->u64_type();
      auto val_u64 = ir_->cast(u64_t, operand_val);
      auto thirty_two_u64 = ir_->uint_immediate_number(u64_t, 32);
      auto hi_u64 = ir_->make_value(spv::OpShiftRightLogical, u64_t, val_u64, thirty_two_u64);
      auto hi = ir_->cast(i32_t, hi_u64);
      auto lo = ir_->cast(i32_t, val_u64);
      auto lo_lsb = ir_->call_glsl450(i32_t, FindILsb_id, lo);
      auto hi_lsb = ir_->call_glsl450(i32_t, FindILsb_id, hi);
      auto thirty_three_i32 = ir_->int_immediate_number(i32_t, 33);
      auto lo_plus_one = ir_->add(lo_lsb, one_i32);
      auto hi_plus_thirty_three = ir_->add(hi_lsb, thirty_three_i32);
      auto lo_zero = ir_->eq(lo, zero_i32);
      auto hi_zero = ir_->eq(hi, zero_i32);
      auto both_zero = ir_->logical_and(lo_zero, hi_zero);
      auto half_pos = ir_->select(lo_zero, hi_plus_thirty_three, lo_plus_one);
      ffs_i32 = ir_->select(both_zero, zero_i32, half_pos);
    } else if (data_type_bits(src_dt) == 32) {
      // Cast operand to i32 so FindILsb's result type matches our i32 arithmetic. For i32 input this is a
      // no-op; for u32 input cast() emits an OpBitcast.
      auto val_i32 = ir_->cast(i32_t, operand_val);
      auto lsb = ir_->call_glsl450(i32_t, FindILsb_id, val_i32);
      auto lsb_plus_one = ir_->add(lsb, one_i32);
      auto is_zero = ir_->eq(val_i32, zero_i32);
      ffs_i32 = ir_->select(is_zero, zero_i32, lsb_plus_one);
    } else {
      QD_NOT_IMPLEMENTED
    }
    // dst_type is i32 across every backend (set by type_check for popcnt / clz / ffs), so this cast is a
    // no-op for the i32 result we just computed.
    val = ir_->cast(dst_type, ffs_i32);
  }
#define UNARY_OP_TO_SPIRV(op, instruction, instruction_id, max_bits)                           \
  else if (stmt->op_type == UnaryOpType::op) {                                                 \
    const uint32_t instruction = instruction_id;                                               \
    if (is_real(src_dt)) {                                                                     \
      if (data_type_bits(src_dt) > max_bits) {                                                 \
        QD_ERROR("Instruction {}({}) does not {}bits operation", #instruction, instruction_id, \
                 data_type_bits(src_dt));                                                      \
      }                                                                                        \
      val = ir_->call_glsl450(src_type, instruction, operand_val);                             \
    } else {                                                                                   \
      QD_NOT_IMPLEMENTED                                                                       \
    }                                                                                          \
  }
  UNARY_OP_TO_SPIRV(round, Round, 1, 64)
  UNARY_OP_TO_SPIRV(floor, Floor, 8, 64)
  UNARY_OP_TO_SPIRV(ceil, Ceil, 9, 64)
  UNARY_OP_TO_SPIRV(sin, Sin, 13, 32)
  UNARY_OP_TO_SPIRV(asin, Asin, 16, 32)
  UNARY_OP_TO_SPIRV(cos, Cos, 14, 32)
  UNARY_OP_TO_SPIRV(acos, Acos, 17, 32)
  UNARY_OP_TO_SPIRV(tan, Tan, 15, 32)
  UNARY_OP_TO_SPIRV(tanh, Tanh, 21, 32)
  UNARY_OP_TO_SPIRV(exp, Exp, 27, 32)
  UNARY_OP_TO_SPIRV(log, Log, 28, 32)
  UNARY_OP_TO_SPIRV(sqrt, Sqrt, 31, 64)
#undef UNARY_OP_TO_SPIRV
  else {
    QD_NOT_IMPLEMENTED
  }
  // For FP-producing unary ops, decorate the result with `NoContraction` when `precise` is set. This is meaningful on
  // actual arithmetic instructions (`OpFNegate` from `neg`, `OpFDiv` synthesized by `inv`) where SPIRV-Cross maps it to
  // MSL's `precise` qualifier. For transcendentals emitted via `OpExtInst GLSL.std.450 Sin/Cos/Log/Sqrt/...`, the
  // SPIR-V spec scopes `NoContraction` to arithmetic instructions so most consumers will ignore it - there is no
  // standard SPIR-V mechanism to force correctly-rounded transcendentals, so on those paths we rely on the driver's
  // default (non-fast-math) stdlib being accurate enough. The decoration is kept as best-effort future-proofing.
  if (stmt->precise && is_real(stmt->element_type())) {
    ir_->maybe_no_contraction(val, /*precise=*/true);
  }
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::generate_overflow_branch(const spirv::Value &cond_v, const std::string &op, const std::string &tb) {
  spirv::Value cond = ir_->ne(cond_v, ir_->cast(cond_v.stype, ir_->const_i32_zero_));
  spirv::Label then_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();
  ir_->make_inst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
  // then block
  ir_->start_label(then_label);
  // `bin->get_tb()` carries the Python traceback that surfaced the binary op - file path, line number,
  // and a copy of the source line - and we want it in the runtime diagnostic. But the SPIR-V debug-printf
  // format string flows verbatim into MoltenVK's SPIRV-Cross -> MSL translator, which embeds it as an MSL
  // string literal; a `"`, `\n`, or `\r` terminates the literal mid-parse and the downstream MSL compile
  // fails with `use of undeclared identifier '<path fragment>'`. A raw `%` is equally hazardous: the
  // concatenated string is the printf-style format with an empty args vector, so an unescaped `%` in the
  // source line (e.g. `a % b`, `"%d" % x`, a `%20` URL-escape) surfaces as a format specifier with no
  // matching argument - undefined behaviour on the validation-layer debug-printf path and on MoltenVK's
  // MSL translation. Escape all four so the printed traceback is preserved byte-for-byte on native Vulkan
  // drivers and still round-trips cleanly through MSL on Apple Silicon. The `%` handling mirrors
  // `sanitize_format_string` above.
  std::string safe_tb;
  safe_tb.reserve(tb.size());
  for (char c : tb) {
    if (c == '"') {
      safe_tb += "\\\"";
    } else if (c == '\n' || c == '\r') {
      safe_tb += ' ';
    } else if (c == '%') {
      safe_tb += "%%";
    } else {
      safe_tb += c;
    }
  }
  ir_->call_debugprintf(op + " overflow detected in " + safe_tb, {});
  ir_->make_inst(spv::OpBranch, merge_label);
  // merge label
  ir_->start_label(merge_label);
}

spirv::Value TaskCodegen::generate_uadd_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  std::vector<std::tuple<spirv::SType, std::string, size_t>> struct_components_;
  struct_components_.emplace_back(a.stype, "result", 0);
  struct_components_.emplace_back(a.stype, "carry", ir_->get_primitive_type_size(a.stype.dt));
  auto struct_type = ir_->create_struct_type(struct_components_);
  auto add_carry = ir_->make_value(spv::OpIAddCarry, struct_type, a, b);
  auto result = ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 0);
  auto carry = ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 1);
  generate_overflow_branch(carry, "Addition", tb);
  return result;
}

spirv::Value TaskCodegen::generate_usub_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  std::vector<std::tuple<spirv::SType, std::string, size_t>> struct_components_;
  struct_components_.emplace_back(a.stype, "result", 0);
  struct_components_.emplace_back(a.stype, "borrow", ir_->get_primitive_type_size(a.stype.dt));
  auto struct_type = ir_->create_struct_type(struct_components_);
  auto add_carry = ir_->make_value(spv::OpISubBorrow, struct_type, a, b);
  auto result = ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 0);
  auto borrow = ir_->make_value(spv::OpCompositeExtract, a.stype, add_carry, 1);
  generate_overflow_branch(borrow, "Subtraction", tb);
  return result;
}

spirv::Value TaskCodegen::generate_sadd_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  // overflow iff (sign(a) == sign(b)) && (sign(a) != sign(result))
  auto result = ir_->make_value(spv::OpIAdd, a.stype, a, b);
  auto zero = ir_->int_immediate_number(a.stype, 0);
  auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
  auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
  auto r_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), result, zero);
  auto a_eq_b = ir_->make_value(spv::OpLogicalEqual, ir_->bool_type(), a_sign, b_sign);
  auto a_neq_r = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(), a_sign, r_sign);
  auto overflow = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_eq_b, a_neq_r);
  generate_overflow_branch(overflow, "Addition", tb);
  return result;
}

spirv::Value TaskCodegen::generate_ssub_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  // overflow iff (sign(a) != sign(b)) && (sign(a) != sign(result))
  auto result = ir_->make_value(spv::OpISub, a.stype, a, b);
  auto zero = ir_->int_immediate_number(a.stype, 0);
  auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
  auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
  auto r_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), result, zero);
  auto a_neq_b = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(), a_sign, b_sign);
  auto a_neq_r = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(), a_sign, r_sign);
  auto overflow = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_neq_b, a_neq_r);
  generate_overflow_branch(overflow, "Subtraction", tb);
  return result;
}

spirv::Value TaskCodegen::generate_umul_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  // overflow iff high bits != 0
  std::vector<std::tuple<spirv::SType, std::string, size_t>> struct_components_;
  struct_components_.emplace_back(a.stype, "low", 0);
  struct_components_.emplace_back(a.stype, "high", ir_->get_primitive_type_size(a.stype.dt));
  auto struct_type = ir_->create_struct_type(struct_components_);
  auto mul_ext = ir_->make_value(spv::OpUMulExtended, struct_type, a, b);
  auto low = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 0);
  auto high = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 1);
  generate_overflow_branch(high, "Multiplication", tb);
  return low;
}

spirv::Value TaskCodegen::generate_smul_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  // overflow if high bits are not all sign bit (0 if positive, -1 if
  // negative) or the sign bit of the low bits is not the expected sign bit.
  std::vector<std::tuple<spirv::SType, std::string, size_t>> struct_components_;
  struct_components_.emplace_back(a.stype, "low", 0);
  struct_components_.emplace_back(a.stype, "high", ir_->get_primitive_type_size(a.stype.dt));
  auto struct_type = ir_->create_struct_type(struct_components_);
  auto mul_ext = ir_->make_value(spv::OpSMulExtended, struct_type, a, b);
  auto low = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 0);
  auto high = ir_->make_value(spv::OpCompositeExtract, a.stype, mul_ext, 1);
  auto zero = ir_->int_immediate_number(a.stype, 0);
  auto minus_one = ir_->int_immediate_number(a.stype, -1);
  auto a_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), a, zero);
  auto b_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), b, zero);
  auto a_not_zero = ir_->ne(a, zero);
  auto b_not_zero = ir_->ne(b, zero);
  auto a_b_not_zero = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), a_not_zero, b_not_zero);
  auto low_sign = ir_->make_value(spv::OpSLessThan, ir_->bool_type(), low, zero);
  auto expected_sign = ir_->make_value(spv::OpLogicalNotEqual, ir_->bool_type(), a_sign, b_sign);
  expected_sign = ir_->make_value(spv::OpLogicalAnd, ir_->bool_type(), expected_sign, a_b_not_zero);
  auto not_expected_sign = ir_->ne(low_sign, expected_sign);
  auto expected_high = ir_->select(expected_sign, minus_one, zero);
  auto not_expected_high = ir_->ne(high, expected_high);
  auto overflow = ir_->make_value(spv::OpLogicalOr, ir_->bool_type(), not_expected_high, not_expected_sign);
  generate_overflow_branch(overflow, "Multiplication", tb);
  return low;
}

spirv::Value TaskCodegen::generate_ushl_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  // overflow iff a << b >> b != a
  auto result = ir_->make_value(spv::OpShiftLeftLogical, a.stype, a, b);
  auto restore = ir_->make_value(spv::OpShiftRightLogical, a.stype, result, b);
  auto overflow = ir_->ne(a, restore);
  generate_overflow_branch(overflow, "Shift left", tb);
  return result;
}

spirv::Value TaskCodegen::generate_sshl_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb) {
  // overflow iff a << b >> b != a
  auto result = ir_->make_value(spv::OpShiftLeftLogical, a.stype, a, b);
  auto restore = ir_->make_value(spv::OpShiftRightArithmetic, a.stype, result, b);
  auto overflow = ir_->ne(a, restore);
  generate_overflow_branch(overflow, "Shift left", tb);
  return result;
}

void TaskCodegen::visit(BinaryOpStmt *bin) {
  const auto lhs_name = bin->lhs->raw_name();
  const auto rhs_name = bin->rhs->raw_name();
  const auto bin_name = bin->raw_name();
  const auto op_type = bin->op_type;

  spirv::SType dst_type = ir_->get_primitive_type(bin->element_type());
  spirv::Value lhs_value = ir_->query_value(lhs_name);
  spirv::Value rhs_value = ir_->query_value(rhs_name);
  spirv::Value bin_value = spirv::Value();

  QD_WARN_IF(lhs_value.stype.id != rhs_value.stype.id, "${} type {} != ${} type {}\n{}", lhs_name,
             lhs_value.stype.dt->to_string(), rhs_name, rhs_value.stype.dt->to_string(), bin->get_tb());

  bool debug = caps_->get(DeviceCapability::spirv_has_non_semantic_info);

  if (debug && op_type == BinaryOpType::add && is_integral(dst_type.dt)) {
    if (is_unsigned(dst_type.dt)) {
      bin_value = generate_uadd_overflow(lhs_value, rhs_value, bin->get_tb());
    } else {
      bin_value = generate_sadd_overflow(lhs_value, rhs_value, bin->get_tb());
    }
    bin_value = ir_->cast(dst_type, bin_value);
  } else if (debug && op_type == BinaryOpType::sub && is_integral(dst_type.dt)) {
    if (is_unsigned(dst_type.dt)) {
      bin_value = generate_usub_overflow(lhs_value, rhs_value, bin->get_tb());
    } else {
      bin_value = generate_ssub_overflow(lhs_value, rhs_value, bin->get_tb());
    }
    bin_value = ir_->cast(dst_type, bin_value);
  } else if (debug && op_type == BinaryOpType::mul && is_integral(dst_type.dt)) {
    if (is_unsigned(dst_type.dt)) {
      bin_value = generate_umul_overflow(lhs_value, rhs_value, bin->get_tb());
    } else {
      bin_value = generate_smul_overflow(lhs_value, rhs_value, bin->get_tb());
    }
    bin_value = ir_->cast(dst_type, bin_value);
  }
  // `bin->precise` is deliberately not threaded into the builder calls below; the post-hoc block at the end of
  // visit(BinaryOpStmt*) is the single source of truth for `NoContraction` decoration, so threading it here would
  // emit a duplicate OpDecorate on the same arithmetic result ID when the subsequent cast is a no-op.
#define BINARY_OP_TO_SPIRV_ARTHIMATIC(op, func)  \
  else if (op_type == BinaryOpType::op) {        \
    bin_value = ir_->func(lhs_value, rhs_value); \
    bin_value = ir_->cast(dst_type, bin_value);  \
  }

  BINARY_OP_TO_SPIRV_ARTHIMATIC(add, add)
  BINARY_OP_TO_SPIRV_ARTHIMATIC(sub, sub)
  BINARY_OP_TO_SPIRV_ARTHIMATIC(mul, mul)
  BINARY_OP_TO_SPIRV_ARTHIMATIC(div, div)
  BINARY_OP_TO_SPIRV_ARTHIMATIC(mod, mod)
#undef BINARY_OP_TO_SPIRV_ARTHIMATIC

#define BINARY_OP_TO_SPIRV_BITWISE(op, sym)                                \
  else if (op_type == BinaryOpType::op) {                                  \
    bin_value = ir_->make_value(spv::sym, dst_type, lhs_value, rhs_value); \
  }

  else if (debug && op_type == BinaryOpType::bit_shl) {
    if (is_unsigned(dst_type.dt)) {
      bin_value = generate_ushl_overflow(lhs_value, rhs_value, bin->get_tb());
    } else {
      bin_value = generate_sshl_overflow(lhs_value, rhs_value, bin->get_tb());
    }
  }
  BINARY_OP_TO_SPIRV_BITWISE(bit_and, OpBitwiseAnd)
  BINARY_OP_TO_SPIRV_BITWISE(bit_or, OpBitwiseOr)
  BINARY_OP_TO_SPIRV_BITWISE(bit_xor, OpBitwiseXor)
  BINARY_OP_TO_SPIRV_BITWISE(bit_shl, OpShiftLeftLogical)
  // NOTE: `OpShiftRightArithmetic` will treat the first bit as sign bit even
  // it's the unsigned type
  else if (op_type == BinaryOpType::bit_sar) {
    bin_value = ir_->make_value(is_unsigned(dst_type.dt) ? spv::OpShiftRightLogical : spv::OpShiftRightArithmetic,
                                dst_type, lhs_value, rhs_value);
  }
#undef BINARY_OP_TO_SPIRV_BITWISE

#define BINARY_OP_TO_SPIRV_LOGICAL(op, func)     \
  else if (op_type == BinaryOpType::op) {        \
    bin_value = ir_->func(lhs_value, rhs_value); \
    bin_value = ir_->cast(dst_type, bin_value);  \
  }

  BINARY_OP_TO_SPIRV_LOGICAL(cmp_lt, lt)
  BINARY_OP_TO_SPIRV_LOGICAL(cmp_le, le)
  BINARY_OP_TO_SPIRV_LOGICAL(cmp_gt, gt)
  BINARY_OP_TO_SPIRV_LOGICAL(cmp_ge, ge)
  BINARY_OP_TO_SPIRV_LOGICAL(cmp_eq, eq)
  BINARY_OP_TO_SPIRV_LOGICAL(cmp_ne, ne)
  BINARY_OP_TO_SPIRV_LOGICAL(logical_and, logical_and)
  BINARY_OP_TO_SPIRV_LOGICAL(logical_or, logical_or)
#undef BINARY_OP_TO_SPIRV_LOGICAL

#define FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(op, instruction, instruction_id, max_bits)                            \
  else if (op_type == BinaryOpType::op) {                                                                         \
    const uint32_t instruction = instruction_id;                                                                  \
    if (is_real(bin->element_type())) {                                                                           \
      if (data_type_bits(bin->element_type()) > max_bits) {                                                       \
        QD_ERROR("[glsl450] the operand type of instruction {}({}) must <= {}bits", #instruction, instruction_id, \
                 max_bits);                                                                                       \
      }                                                                                                           \
      bin_value = ir_->call_glsl450(dst_type, instruction, lhs_value, rhs_value);                                 \
    } else {                                                                                                      \
      QD_NOT_IMPLEMENTED                                                                                          \
    }                                                                                                             \
  }

  FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(atan2, Atan2, 25, 32)
  FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC(pow, Pow, 26, 32)
#undef FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC

#define BINARY_OP_TO_SPIRV_FUNC(op, S_inst, S_inst_id, U_inst, U_inst_id, F_inst, F_inst_id) \
  else if (op_type == BinaryOpType::op) {                                                    \
    const uint32_t S_inst = S_inst_id;                                                       \
    const uint32_t U_inst = U_inst_id;                                                       \
    const uint32_t F_inst = F_inst_id;                                                       \
    const auto dst_dt = bin->element_type();                                                 \
    if (is_integral(dst_dt)) {                                                               \
      if (is_signed(dst_dt)) {                                                               \
        bin_value = ir_->call_glsl450(dst_type, S_inst, lhs_value, rhs_value);               \
      } else {                                                                               \
        bin_value = ir_->call_glsl450(dst_type, U_inst, lhs_value, rhs_value);               \
      }                                                                                      \
    } else if (is_real(dst_dt)) {                                                            \
      bin_value = ir_->call_glsl450(dst_type, F_inst, lhs_value, rhs_value);                 \
    } else {                                                                                 \
      QD_NOT_IMPLEMENTED                                                                     \
    }                                                                                        \
  }

  BINARY_OP_TO_SPIRV_FUNC(min, SMin, 39, UMin, 38, FMin, 37)
  BINARY_OP_TO_SPIRV_FUNC(max, SMax, 42, UMax, 41, FMax, 40)
#undef BINARY_OP_TO_SPIRV_FUNC
  else if (op_type == BinaryOpType::truediv) {
    lhs_value = ir_->cast(dst_type, lhs_value);
    rhs_value = ir_->cast(dst_type, rhs_value);
    // As with the arithmetic macro above, leave decoration to the post-hoc block.
    bin_value = ir_->div(lhs_value, rhs_value);
  }
  else {
    QD_NOT_IMPLEMENTED;
  }
  // Single source of truth for `NoContraction` on FP-producing binary ops. Covers:
  //   - arithmetic (add/sub/mul/div/mod/truediv): the intervening `ir_->cast(dst_type, bin_value)` is a no-op in the
  //     common post-type_check case where operand type already matches `dst_type`, so this decorates the
  //     OpF{Add,Sub,...} itself; in the rare non-no-op case it decorates the FConvert, which per spec drops the
  //     decoration silently.
  //   - FP binary transcendentals (atan2, pow): emitted by `FLOAT_BINARY_OP_TO_SPIRV_FLOAT_FUNC` through
  //     `ir_->call_glsl450(...)` with no internal `maybe_no_contraction`; SPIR-V scopes `NoContraction` to arithmetic
  //     instructions so most consumers ignore it on `OpExtInst`, but the decoration is best-effort future-proofing and
  //     should be applied uniformly with the unary transcendental path.
  // Do NOT thread `bin->precise` into the builder calls above; the builders would then emit a duplicate OpDecorate on
  // the same result ID.
  if (bin->precise && is_real(bin->element_type())) {
    ir_->maybe_no_contraction(bin_value, /*precise=*/true);
  }
  ir_->register_value(bin_name, bin_value);
}

void TaskCodegen::visit(TernaryOpStmt *tri) {
  QD_ASSERT(tri->op_type == TernaryOpType::select);
  spirv::Value op1 = ir_->query_value(tri->op1->raw_name());
  spirv::Value op2 = ir_->query_value(tri->op2->raw_name());
  spirv::Value op3 = ir_->query_value(tri->op3->raw_name());
  spirv::Value tri_val =
      ir_->cast(ir_->get_primitive_type(tri->element_type()), ir_->select(ir_->cast(ir_->bool_type(), op1), op2, op3));
  ir_->register_value(tri->raw_name(), tri_val);
}

inline bool TaskCodegen::ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void TaskCodegen::visit(InternalFuncStmt *stmt) {
  spirv::Value val;

  // Note: the SPIR-V-only `subgroupAdd` / `subgroupMul` / `subgroupMin` / `subgroupMax` / `subgroupAnd` /
  // `subgroupOr` / `subgroupXor` reductions have been removed.  Likewise the
  // `subgroupInclusive{Add,Mul,Min,Max,And,Or,Xor}` ops are gone: all seven are implemented as portable ``@qd.func``
  // Hillis-Steele scans over `subgroupShuffleUp` in Python, so the SPIR-V codegen branch and the matching internal-op
  // registrations have been removed.  An ``InternalFuncStmt`` carrying one of those removed names would fall through
  // the dispatcher below and hit the final ``QD_ERROR``, surfacing the mismatch instead of registering a
  // default-constructed ``spirv::Value`` and producing invalid SPIR-V at run time.

  const std::unordered_set<std::string> shuffle_ops{"subgroupShuffleDown", "subgroupShuffleUp", "subgroupShuffle"};

  if (stmt->func_name == "composite_extract_0") {
    val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(), ir_->query_value(stmt->args[0]->raw_name()), 0);
  } else if (stmt->func_name == "composite_extract_1") {
    val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(), ir_->query_value(stmt->args[0]->raw_name()), 1);
  } else if (stmt->func_name == "composite_extract_2") {
    val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(), ir_->query_value(stmt->args[0]->raw_name()), 2);
  } else if (stmt->func_name == "composite_extract_3") {
    val = ir_->make_value(spv::OpCompositeExtract, ir_->f32_type(), ir_->query_value(stmt->args[0]->raw_name()), 3);
  } else if (stmt->func_name == "workgroupBarrier") {
    ir_->make_inst(spv::OpControlBarrier, ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
                   ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
                   ir_->int_immediate_number(ir_->i32_type(), spv::MemorySemanticsWorkgroupMemoryMask |
                                                                  spv::MemorySemanticsAcquireReleaseMask));
    val = ir_->const_i32_zero_;
  } else if (stmt->func_name == "localInvocationId") {
    val = ir_->cast(ir_->i32_type(), ir_->get_local_invocation_id(0));
  } else if (stmt->func_name == "globalInvocationId") {
    val = ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0));
  } else if (stmt->func_name == "workgroupMemoryBarrier") {
    ir_->make_inst(spv::OpMemoryBarrier, ir_->int_immediate_number(ir_->i32_type(), spv::ScopeWorkgroup),
                   ir_->int_immediate_number(ir_->i32_type(), spv::MemorySemanticsWorkgroupMemoryMask |
                                                                  spv::MemorySemanticsAcquireReleaseMask));
    val = ir_->const_i32_zero_;
  } else if (stmt->func_name == "gridMemoryBarrier") {
    // Device-scope memory fence (orders memory ops across the entire grid). On Vulkan this lowers to an
    // `OpMemoryBarrier(ScopeDevice, ...)` with acquire-release semantics on workgroup + uniform (= storage-buffer in
    // Vulkan's mapping) memory; that is sufficient for the canonical use cases (decoupled look-back scan, inter-block
    // flag publishing). MoltenVK translates this to MSL `atomic_thread_fence(metal::memory_scope_device)` (MSL 2.0+,
    // macOS 10.13+ / iOS 11+); on pre-A11 Apple GPUs / very old macOS Intel GPUs the cross-workgroup ordering
    // guarantees are weaker and users should validate empirically.
    ir_->make_inst(spv::OpMemoryBarrier, ir_->int_immediate_number(ir_->i32_type(), spv::ScopeDevice),
                   ir_->int_immediate_number(ir_->i32_type(), spv::MemorySemanticsUniformMemoryMask |
                                                                  spv::MemorySemanticsWorkgroupMemoryMask |
                                                                  spv::MemorySemanticsAcquireReleaseMask));
    val = ir_->const_i32_zero_;
  } else if (stmt->func_name == "subgroupElect") {
    val = ir_->make_value(spv::OpGroupNonUniformElect, ir_->bool_type(),
                          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup));
    val = ir_->cast(ir_->i32_type(), val);
  } else if (stmt->func_name == "subgroupBarrier") {
    ir_->make_inst(spv::OpControlBarrier, ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
                   ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), ir_->const_i32_zero_);
    val = ir_->const_i32_zero_;
  } else if (stmt->func_name == "subgroupMemoryBarrier") {
    // The Memory Semantics operand of OpMemoryBarrier must include both an ordering bit (Acquire / Release /
    // AcquireRelease / SequentiallyConsistent) and at least one storage class (UniformMemory / WorkgroupMemory /
    // ImageMemory / ...).  The previous emission used 0 for Semantics, which is invalid SPIR-V and behaves as a no-op
    // on drivers that accept it.  Match the pattern used for `workgroupMemoryBarrier` above: AcquireRelease with the
    // storage classes Quadrants kernels actually touch (uniform buffers + workgroup-shared memory).
    ir_->make_inst(spv::OpMemoryBarrier, ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup),
                   ir_->int_immediate_number(ir_->i32_type(), spv::MemorySemanticsUniformMemoryMask |
                                                                  spv::MemorySemanticsWorkgroupMemoryMask |
                                                                  spv::MemorySemanticsAcquireReleaseMask));
    val = ir_->const_i32_zero_;
  } else if (stmt->func_name == "subgroupInvocationId") {
    val = ir_->cast(ir_->i32_type(), ir_->get_subgroup_invocation_id());
  } else if (stmt->func_name == "subgroupBroadcast") {
    auto value = ir_->query_value(stmt->args[0]->raw_name());
    auto index = ir_->query_value(stmt->args[1]->raw_name());
    val = ir_->make_value(spv::OpGroupNonUniformBroadcast, value.stype,
                          ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), value, index);
  } else if (stmt->func_name == "subgroupBallotU32") {
    // ``OpGroupNonUniformBallot`` produces a uvec4 of 128 ballot bits.  Component 0 covers lanes 0..31, which is
    // exactly what the ``u32`` ballot form ( ``ballot_first_n``) advertises; lanes 32..63 (on wave64 backends) are not
    // represented in the u32 result, matching the AMDGPU / CUDA u32 forms.
    auto predicate = ir_->query_value(stmt->args[0]->raw_name());
    auto pred_bool =
        ir_->make_value(spv::OpINotEqual, ir_->bool_type(), predicate, ir_->int_immediate_number(ir_->i32_type(), 0));
    auto ballot_vec = ir_->make_value(spv::OpGroupNonUniformBallot, ir_->v4_u32_type(),
                                      ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), pred_bool);
    val = ir_->make_value(spv::OpCompositeExtract, ir_->u32_type(), ballot_vec, 0);
  } else if (stmt->func_name == "subgroupBallotU64") {
    // For the full-subgroup u64 form we extract components 0 and 1 (lanes 0..31 and 32..63 respectively) and pack
    // them into a single u64: ``u64(hi) << 32 | u64(lo)``.  On wave32 component 1 is naturally zero (no lanes 32+
    // exist), so the high half of the result is zero and the API is uniform across wavefront modes.
    auto predicate = ir_->query_value(stmt->args[0]->raw_name());
    auto pred_bool =
        ir_->make_value(spv::OpINotEqual, ir_->bool_type(), predicate, ir_->int_immediate_number(ir_->i32_type(), 0));
    auto ballot_vec = ir_->make_value(spv::OpGroupNonUniformBallot, ir_->v4_u32_type(),
                                      ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), pred_bool);
    auto lo = ir_->make_value(spv::OpCompositeExtract, ir_->u32_type(), ballot_vec, 0);
    auto hi = ir_->make_value(spv::OpCompositeExtract, ir_->u32_type(), ballot_vec, 1);
    auto lo64 = ir_->cast(ir_->u64_type(), lo);
    auto hi64 = ir_->cast(ir_->u64_type(), hi);
    auto shift = ir_->uint_immediate_number(ir_->u64_type(), 32u);
    auto hi_shifted = ir_->make_value(spv::OpShiftLeftLogical, ir_->u64_type(), hi64, shift);
    val = ir_->make_value(spv::OpBitwiseOr, ir_->u64_type(), lo64, hi_shifted);
  } else if (shuffle_ops.find(stmt->func_name) != shuffle_ops.end()) {
    auto arg0 = ir_->query_value(stmt->args[0]->raw_name());
    auto arg1 = ir_->query_value(stmt->args[1]->raw_name());
    auto stype = ir_->get_primitive_type(stmt->args[0]->ret_type);
    spv::Op spv_op;

    if (ends_with(stmt->func_name, "Down")) {
      spv_op = spv::OpGroupNonUniformShuffleDown;
    } else if (ends_with(stmt->func_name, "Up")) {
      spv_op = spv::OpGroupNonUniformShuffleUp;
    } else if (ends_with(stmt->func_name, "Shuffle")) {
      spv_op = spv::OpGroupNonUniformShuffle;
    } else {
      QD_ERROR("Unsupported operation: {}", stmt->func_name);
    }

    val = ir_->make_value(spv_op, stype, ir_->int_immediate_number(ir_->i32_type(), spv::ScopeSubgroup), arg0, arg1);
  } else if (stmt->func_name == "spirv_clock_i64") {
    // OpReadClockKHR returns a 64-bit unsigned integer
    // Scope: Device (1) for device-wide clock
    if (caps_->get(DeviceCapability::spirv_has_shader_clock) && caps_->get(DeviceCapability::spirv_has_int64)) {
      spirv::Value clock_val = ir_->make_value(spv::OpReadClockKHR, ir_->u64_type(),
                                               ir_->uint_immediate_number(ir_->u32_type(), spv::ScopeDevice));
      // Cast u64 to i64 as the return type is i64
      val = ir_->make_value(spv::OpBitcast, ir_->i64_type(), clock_val);
    } else {
      // Return 0 if shader clock is not supported
      val = ir_->int_immediate_number(ir_->i64_type(), 0);
    }
  } else {
    QD_ERROR("Unsupported InternalFuncStmt for SPIR-V codegen: {}", stmt->func_name);
  }
  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(AtomicOpStmt *stmt) {
  const auto dt = stmt->dest->element_type().ptr_removed();

  spirv::Value data = ir_->query_value(stmt->val->raw_name());
  spirv::Value val;
  bool use_subgroup_reduction = false;

  if (stmt->is_reduction && caps_->get(DeviceCapability::spirv_has_subgroup_arithmetic)) {
    spv::Op atomic_op = spv::OpNop;
    bool negation = false;
    if (is_integral(dt)) {
      if (stmt->op_type == AtomicOpType::add) {
        atomic_op = spv::OpGroupIAdd;
      } else if (stmt->op_type == AtomicOpType::sub) {
        atomic_op = spv::OpGroupIAdd;
        negation = true;
      } else if (stmt->op_type == AtomicOpType::min) {
        atomic_op = is_signed(dt) ? spv::OpGroupSMin : spv::OpGroupUMin;
      } else if (stmt->op_type == AtomicOpType::max) {
        atomic_op = is_signed(dt) ? spv::OpGroupSMax : spv::OpGroupUMax;
      }
    } else if (is_real(dt)) {
      if (stmt->op_type == AtomicOpType::add) {
        atomic_op = spv::OpGroupFAdd;
      } else if (stmt->op_type == AtomicOpType::sub) {
        atomic_op = spv::OpGroupFAdd;
        negation = true;
      } else if (stmt->op_type == AtomicOpType::min) {
        atomic_op = spv::OpGroupFMin;
      } else if (stmt->op_type == AtomicOpType::max) {
        atomic_op = spv::OpGroupFMax;
      }
    }

    if (atomic_op != spv::OpNop) {
      spirv::Value scope_subgroup = ir_->int_immediate_number(ir_->i32_type(), 3);
      spirv::Value operation_reduce = ir_->const_i32_zero_;
      if (negation) {
        if (is_integral(dt)) {
          data = ir_->make_value(spv::OpSNegate, data.stype, data);
        } else {
          data = ir_->make_value(spv::OpFNegate, data.stype, data);
        }
      }
      data = ir_->make_value(atomic_op, ir_->get_primitive_type(dt), scope_subgroup, operation_reduce, data);
      val = data;
      use_subgroup_reduction = true;
    }
  }

  spirv::Label then_label;
  spirv::Label merge_label;

  if (use_subgroup_reduction) {
    spirv::Value subgroup_id = ir_->get_subgroup_invocation_id();
    spirv::Value cond = ir_->make_value(spv::OpIEqual, ir_->bool_type(), subgroup_id, ir_->const_i32_zero_);

    then_label = ir_->new_label();
    merge_label = ir_->new_label();
    ir_->make_inst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
    ir_->start_label(then_label);
  }

  spirv::Value addr_ptr;
  spirv::Value dest_val = ir_->query_value(stmt->dest->raw_name());
  // Shared arrays already have a pointer from OpAccessChain (dest_is_ptr=true).
  // at_buffer() looks up ptr_to_buffers_ to find the StorageBuffer and compute
  // a byte offset - shared/workgroup arrays aren't in ptr_to_buffers_, so
  // at_buffer() would fail on them.
  const bool dest_is_ptr = dest_val.stype.flag == TypeKind::kPtr;
  // The native-add branches originally called at_buffer() directly, but shared
  // arrays can now reach this path, so all branches need the dest_is_ptr guard.
  if (dt->is_primitive(PrimitiveTypeID::f64)) {
    if (caps_->get(DeviceCapability::spirv_has_atomic_float64_add) && stmt->op_type == AtomicOpType::add) {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, dt);
    } else {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, ir_->get_quadrants_uint_type(dt));
    }
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    if (caps_->get(DeviceCapability::spirv_has_atomic_float_add) && stmt->op_type == AtomicOpType::add) {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, dt);
    } else {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, ir_->get_quadrants_uint_type(dt));
    }
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // f16 needs the same uint-typed pointer as f32/f64 for the CAS path.
    // Without this, at_buffer returns pointer-to-f16 but the CAS loop uses
    // OpAtomicLoad(u16, ...) causing a SPIR-V type mismatch.
    if (caps_->get(DeviceCapability::spirv_has_atomic_float16_add) && stmt->op_type == AtomicOpType::add) {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, dt);
    } else {
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, ir_->get_quadrants_uint_type(dt));
    }
  } else {
    addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, dt);
  }

  auto ret_type = ir_->get_primitive_type(dt);

  if (is_real(dt)) {
    // Only initialized for add (the only op with native float atomic support).
    // Safe: use_native_atomics is only true when op_type == add.
    spv::Op atomic_fp_op = spv::OpNop;
    if (stmt->op_type == AtomicOpType::add) {
      atomic_fp_op = spv::OpAtomicFAddEXT;
    }

    bool use_native_atomics = false;

    if (dt->is_primitive(PrimitiveTypeID::f64)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float64_add) && stmt->op_type == AtomicOpType::add) {
        use_native_atomics = true;
      }
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float_add) && stmt->op_type == AtomicOpType::add) {
        use_native_atomics = true;
      }
    } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
      if (caps_->get(DeviceCapability::spirv_has_atomic_float16_add) && stmt->op_type == AtomicOpType::add) {
        use_native_atomics = true;
      }
    }
    // The checks above use buffer capabilities. For shared pointers, override
    // with shared capabilities (buffer and shared support are independent).
    if (dest_is_ptr && stmt->op_type == AtomicOpType::add) {
      use_native_atomics = has_native_float_atomic_add(*caps_, dt, true);
    }
    // Uint-retyped shared arrays have a uint pointer - native float atomics
    // would produce invalid SPIR-V on them.
    if (uint_backed_shared_float_ptr_stmts_.count(stmt->dest)) {
      use_native_atomics = false;
    }

    if (stmt->op_type == AtomicOpType::xchg && !dest_is_ptr && !dt->is_primitive(PrimitiveTypeID::f16)) {
      // Float xchg: route through OpAtomicExchange on the uint-backed buffer pointer (already established by
      // the addr_ptr block above when no native float-atomic-add cap is in play). OpAtomicExchange supports
      // float operands per the SPIR-V spec, but going through uint avoids any spirv_has_atomic_float_* cap
      // dependency and works on every backend (including MoltenVK / spirv-cross-msl on Apple Silicon).
      // Shared (workgroup) float xchg and f16 xchg are not yet covered -- they would need uint-backing
      // analogous to the add CAS path; out of scope for the initial xchg landing.
      auto uint_dt = ir_->get_quadrants_uint_type(dt);
      auto uint_ret_type = ir_->get_primitive_type(uint_dt);
      auto uint_data = ir_->make_value(spv::OpBitcast, uint_ret_type, data);
      val = ir_->make_value(spv::OpAtomicExchange, uint_ret_type, addr_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics=*/ir_->const_i32_zero_, uint_data);
      val = ir_->make_value(spv::OpBitcast, ret_type, val);
    } else if (use_native_atomics) {
      val = ir_->make_value(atomic_fp_op, ir_->get_primitive_type(dt), addr_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics=*/ir_->const_i32_zero_, data);
    } else if (dest_is_ptr) {
      // Shared float arrays use uint-backed CAS (width-aware for f16->u32).
      // Integer shared atomics don't need this - they use native OpAtomicIAdd
      // etc. directly on the shared pointer.
      QD_ASSERT_INFO(stmt->op_type != AtomicOpType::xchg,
                     "atomic_exchange on shared (workgroup) float arrays is not yet implemented for SPIR-V; would "
                     "need uint-backing analogous to the shared float-add CAS path");
      val = shared_float_atomic(*ir_, stmt->op_type, addr_ptr, data, dt);
    } else {
      // Global f16 xchg falls through here because the uint-bitcast xchg branch above explicitly excludes f16
      // (it would need a width-mismatched bitcast, same as the f16 atomic-add CAS path). float_atomic itself has
      // no xchg case and would otherwise abort with a generic QD_NOT_IMPLEMENTED -- promote to a clearer message.
      QD_ASSERT_INFO(stmt->op_type != AtomicOpType::xchg,
                     "atomic_exchange on f16 (global memory) is not yet implemented for SPIR-V; would need a "
                     "width-mismatched uint-backed bitcast analogous to the f16 atomic-add CAS path");
      val = ir_->float_atomic(stmt->op_type, addr_ptr, data, dt);
    }
  } else if (is_integral(dt)) {
    if (stmt->op_type == AtomicOpType::cas) {
      // OpAtomicCompareExchange takes (scope, sem_eq, sem_neq, value, comparator) and returns the value
      // originally at `addr_ptr`. We surface that prior value to the user; success is recovered with
      // `(returned == expected)`. Matches CUDA atomicCAS semantics. Uses Relaxed semantics like every other
      // atomic op in this file - if surrounding-memory ordering is needed, the user pairs the CAS with
      // qd.simt.block.mem_fence() / qd.simt.grid.mem_fence() the same way they would for atomic_add.
      QD_ASSERT(stmt->expected != nullptr);
      spirv::Value expected_val = ir_->query_value(stmt->expected->raw_name());
      val = ir_->make_value(spv::OpAtomicCompareExchange, ret_type, addr_ptr,
                            /*scope=*/ir_->const_i32_one_,
                            /*semantics if equal=*/ir_->const_i32_zero_,
                            /*semantics if unequal=*/ir_->const_i32_zero_, data, expected_val);
      ir_->register_value(stmt->raw_name(), val);
      return;
    }
    bool use_native_atomics = false;
    spv::Op op;
    if (stmt->op_type == AtomicOpType::add) {
      op = spv::OpAtomicIAdd;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::sub) {
      op = spv::OpAtomicISub;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::mul) {
      // dest_is_ptr guard needed here too - at_buffer would crash on shared
      // integer arrays (same reason as the float branches above).
      addr_ptr = dest_is_ptr ? dest_val : at_buffer(stmt->dest, ir_->get_quadrants_uint_type(dt));
      val = ir_->integer_atomic(stmt->op_type, addr_ptr, data, dt);
      use_native_atomics = false;
    } else if (stmt->op_type == AtomicOpType::min) {
      op = is_signed(dt) ? spv::OpAtomicSMin : spv::OpAtomicUMin;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::max) {
      op = is_signed(dt) ? spv::OpAtomicSMax : spv::OpAtomicUMax;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::bit_or) {
      op = spv::OpAtomicOr;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::bit_and) {
      op = spv::OpAtomicAnd;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::bit_xor) {
      op = spv::OpAtomicXor;
      use_native_atomics = true;
    } else if (stmt->op_type == AtomicOpType::xchg) {
      op = spv::OpAtomicExchange;
      use_native_atomics = true;
    } else {
      QD_NOT_IMPLEMENTED
    }

    if (use_native_atomics) {
      auto uint_type = ir_->get_primitive_uint_type(dt);

      if (data.stype.id != addr_ptr.stype.element_type_id) {
        data = ir_->make_value(spv::OpBitcast, ret_type, data);
      }

      // Pick the SPIR-V `Scope` and `MemorySemantics` memory-class flag based on the storage class of the atomic's
      // target pointer. Workgroup (shared) memory atomics need `Workgroup` scope and `WorkgroupMemory` semantics;
      // device-buffer atomics need `Device` + `UniformMemory`. Without this distinction MoltenVK / SPIRV-Cross
      // translates a workgroup-storage `OpAtomicOr` with `Device` scope to MSL
      // `atomic_fetch_or_explicit(..., memory_scope_device)` on a `threadgroup atomic_int`, which is mismatched and
      // silently does not update the threadgroup-shared slot. This surfaced as
      // `block.sync_{any,all,count}_nonzero` returning the initialiser value on Metal even though the same emulation
      // works on Vulkan (whose drivers happen to tolerate the over-strong scope).
      const bool is_workgroup = addr_ptr.stype.storage_class == spv::StorageClassWorkgroup;
      const auto scope_const =
          ir_->int_immediate_number(ir_->i32_type(), is_workgroup ? spv::ScopeWorkgroup : spv::ScopeDevice);
      const auto memory_class_mask =
          is_workgroup ? spv::MemorySemanticsWorkgroupMemoryMask : spv::MemorySemanticsUniformMemoryMask;
      ir_->make_inst(
          spv::OpMemoryBarrier, scope_const,
          ir_->uint_immediate_number(ir_->u32_type(), spv::MemorySemanticsAcquireReleaseMask | memory_class_mask));
      val = ir_->make_value(op, ret_type, addr_ptr,
                            /*scope=*/scope_const,
                            /*semantics=*/ir_->const_i32_zero_, data);

      if (val.stype.id != ret_type.id) {
        val = ir_->make_value(spv::OpBitcast, ret_type, val);
      }
    }
  } else {
    QD_NOT_IMPLEMENTED
  }

  if (use_subgroup_reduction) {
    ir_->make_inst(spv::OpBranch, merge_label);
    ir_->start_label(merge_label);
  }

  ir_->register_value(stmt->raw_name(), val);
}

void TaskCodegen::visit(IfStmt *if_stmt) {
  spirv::Value cond_v = ir_->cast(ir_->bool_type(), ir_->query_value(if_stmt->cond->raw_name()));
  spirv::Value cond = ir_->ne(cond_v, ir_->cast(ir_->bool_type(), ir_->const_i32_zero_));
  spirv::Label then_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();
  spirv::Label else_label = ir_->new_label();
  ir_->make_inst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, cond, then_label, else_label);
  // then block
  ir_->start_label(then_label);
  if (if_stmt->true_statements) {
    if_stmt->true_statements->accept(this);
  }
  // ContinueStmt must be in IfStmt
  if (gen_label_) {  // Skip OpBranch, because ContinueStmt already generated
                     // one
    gen_label_ = false;
  } else {
    ir_->make_inst(spv::OpBranch, merge_label);
  }
  // else block
  ir_->start_label(else_label);
  if (if_stmt->false_statements) {
    if_stmt->false_statements->accept(this);
  }
  if (gen_label_) {
    gen_label_ = false;
  } else {
    ir_->make_inst(spv::OpBranch, merge_label);
  }
  // merge label
  ir_->start_label(merge_label);
}

void TaskCodegen::visit(RangeForStmt *for_stmt) {
  auto loop_var_name = for_stmt->raw_name();
  // Must get init label after making value(to make sure they are correct)
  spirv::Label init_label = ir_->current_label();
  spirv::Label head_label = ir_->new_label();
  spirv::Label body_label = ir_->new_label();
  spirv::Label continue_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();

  spirv::Value begin_ = ir_->query_value(for_stmt->begin->raw_name());
  spirv::Value end_ = ir_->query_value(for_stmt->end->raw_name());
  spirv::Value init_value;
  spirv::Value extent_value;
  if (!for_stmt->reversed) {
    init_value = begin_;
    extent_value = end_;
  } else {
    // reversed for loop
    init_value = ir_->sub(end_, ir_->const_i32_one_);
    extent_value = begin_;
  }
  ir_->make_inst(spv::OpBranch, head_label);

  // Loop head
  ir_->start_label(head_label);
  spirv::PhiValue loop_var = ir_->make_phi(init_value.stype, 2);
  loop_var.set_incoming(0, init_value, init_label);
  spirv::Value loop_cond;
  if (!for_stmt->reversed) {
    loop_cond = ir_->lt(loop_var, extent_value);
  } else {
    loop_cond = ir_->ge(loop_var, extent_value);
  }
  ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label, spv::LoopControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, loop_cond, body_label, merge_label);

  // loop body
  ir_->start_label(body_label);
  push_loop_control_labels(continue_label, merge_label);
  ir_->register_value(loop_var_name, spirv::Value(loop_var));
  for_stmt->body->accept(this);
  pop_loop_control_labels();
  ir_->make_inst(spv::OpBranch, continue_label);

  // loop continue
  ir_->start_label(continue_label);
  spirv::Value next_value;
  if (!for_stmt->reversed) {
    next_value = ir_->add(loop_var, ir_->const_i32_one_);
  } else {
    next_value = ir_->sub(loop_var, ir_->const_i32_one_);
  }
  loop_var.set_incoming(1, next_value, ir_->current_label());
  ir_->make_inst(spv::OpBranch, head_label);
  // loop merge
  ir_->start_label(merge_label);
}

void TaskCodegen::visit(WhileStmt *stmt) {
  spirv::Label head_label = ir_->new_label();
  spirv::Label body_label = ir_->new_label();
  spirv::Label continue_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();
  ir_->make_inst(spv::OpBranch, head_label);

  // Loop head
  ir_->start_label(head_label);
  ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label, spv::LoopControlMaskNone);
  ir_->make_inst(spv::OpBranch, body_label);

  // loop body
  ir_->start_label(body_label);
  push_loop_control_labels(continue_label, merge_label);
  stmt->body->accept(this);
  pop_loop_control_labels();
  ir_->make_inst(spv::OpBranch, continue_label);

  // loop continue
  ir_->start_label(continue_label);
  ir_->make_inst(spv::OpBranch, head_label);

  // loop merge
  ir_->start_label(merge_label);
}

void TaskCodegen::visit(WhileControlStmt *stmt) {
  spirv::Value cond_v = ir_->cast(ir_->bool_type(), ir_->query_value(stmt->cond->raw_name()));
  spirv::Value cond = ir_->eq(cond_v, ir_->cast(ir_->bool_type(), ir_->const_i32_zero_));
  spirv::Label then_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();

  ir_->make_inst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
  ir_->start_label(then_label);
  ir_->make_inst(spv::OpBranch, current_merge_label());  // break;
  ir_->start_label(merge_label);
}

void TaskCodegen::visit(ContinueStmt *stmt) {
  auto stmt_in_off_for = [stmt]() {
    QD_ASSERT(stmt->scope != nullptr);
    if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
      QD_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                offl->task_type == OffloadedStmt::TaskType::struct_for);
      return true;
    }
    return false;
  };
  if (stmt_in_off_for()) {
    // Return means end THIS main loop and start next loop, not exit kernel
    ir_->make_inst(spv::OpBranch, return_label());
  } else {
    ir_->make_inst(spv::OpBranch, current_continue_label());
  }
  gen_label_ = true;  // Only ContinueStmt will cause duplicate OpBranch,
                      // which should be eliminated
}

void TaskCodegen::emit_headers() {
  /*
  for (int root = 0; root < compiled_structs_.size(); ++root) {
    get_buffer_value({BufferType::Root, root});
  }
  */
  std::array<int, 3> group_size = {task_attribs_.advisory_num_threads_per_group, 1, 1};
  ir_->set_work_group_size(group_size);
  std::vector<spirv::Value> buffers;
  if (caps_->get(DeviceCapability::spirv_version) > 0x10300) {
    buffers = shared_array_binds_;
    // One buffer can be bound to different bind points but has to be unique
    // in OpEntryPoint interface declarations.
    // From Spec: before SPIR-V version 1.4, duplication of these interface id
    // is tolerated. Starting with version 1.4, an interface id must not
    // appear more than once.
    std::unordered_set<spirv::Value, spirv::ValueHasher> entry_point_values;
    for (const auto &bb : task_attribs_.buffer_binds) {
      for (auto &it : buffer_value_map_) {
        if (it.first.first == bb.buffer) {
          entry_point_values.insert(it.second);
        }
      }
    }
    buffers.insert(buffers.end(), entry_point_values.begin(), entry_point_values.end());
  }
  ir_->commit_kernel_function(kernel_function_, "main", buffers,
                              group_size);  // kernel entry
}

void TaskCodegen::generate_serial_kernel(OffloadedStmt *stmt) {
  task_attribs_.name = task_name_;
  task_attribs_.task_type = OffloadedTaskType::serial;
  task_attribs_.advisory_total_num_threads = 1;
  task_attribs_.advisory_num_threads_per_group = 1;

  // The computation for a single work is wrapped inside a function, so that
  // we can do grid-strided loop.
  ir_->start_function(kernel_function_);
  // Initialise the adstack overflow-signal accumulator before any user code so the zero-init dominates
  // every push site and the task-end read. See `ensure_any_overflow_signal_var` doc for details.
  ensure_any_overflow_signal_var();
  spirv::Value cond = ir_->eq(ir_->get_global_invocation_id(0),
                              ir_->uint_immediate_number(ir_->u32_type(), 0));  // if (gl_GlobalInvocationID.x > 0)
  spirv::Label then_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();
  kernel_return_label_ = merge_label;

  ir_->make_inst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, cond, then_label, merge_label);
  ir_->start_label(then_label);

  // serial kernel
  stmt->body->accept(this);

  ir_->make_inst(spv::OpBranch, merge_label);
  ir_->start_label(merge_label);
  emit_adstack_task_end_overflow_check();
  ir_->make_inst(spv::OpReturn);       // return;
  ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

  task_attribs_.buffer_binds = get_buffer_binds();
}

void TaskCodegen::gen_array_range(Stmt *stmt) {
  /* Fix issue 7493
   *
   * Prevent repeated range generation for the same array
   * when loop range has multiple dimensions.
   */
  if (ir_->check_value_existence(stmt->raw_name())) {
    return;
  }
  int num_operands = stmt->num_operands();
  for (int i = 0; i < num_operands; i++) {
    gen_array_range(stmt->operand(i));
  }
  offload_loop_motion_.insert(stmt);
  stmt->accept(this);
}

void TaskCodegen::generate_range_for_kernel(OffloadedStmt *stmt) {
  task_attribs_.name = task_name_;
  task_attribs_.task_type = OffloadedTaskType::range_for;

  task_attribs_.range_for_attribs = TaskAttributes::RangeForAttributes();
  auto &range_for_attribs = task_attribs_.range_for_attribs.value();
  range_for_attribs.const_begin = stmt->const_begin;
  range_for_attribs.const_end = stmt->const_end;
  range_for_attribs.begin = (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
  range_for_attribs.end = (stmt->const_end ? stmt->end_value : stmt->end_offset);

  ir_->start_function(kernel_function_);
  ensure_any_overflow_signal_var();
  const std::string total_elems_name("total_elems");
  spirv::Value total_elems;
  spirv::Value begin_expr_value;
  if (range_for_attribs.const_range()) {
    const int num_elems = range_for_attribs.end - range_for_attribs.begin;
    begin_expr_value = ir_->int_immediate_number(ir_->i32_type(), stmt->begin_value, false);  // Named Constant
    total_elems = ir_->int_immediate_number(ir_->i32_type(), num_elems,
                                            false);  // Named Constant
    task_attribs_.advisory_total_num_threads = num_elems;
  } else {
    spirv::Value end_expr_value;
    if (stmt->end_stmt) {
      // Range from args
      QD_ASSERT(stmt->const_begin);
      begin_expr_value = ir_->int_immediate_number(ir_->i32_type(), stmt->begin_value, false);
      gen_array_range(stmt->end_stmt);
      end_expr_value = ir_->query_value(stmt->end_stmt->raw_name());
    } else {
      // Range from gtmp / constant
      if (!stmt->const_begin) {
        spirv::Value begin_idx = ir_->make_value(spv::OpShiftRightArithmetic, ir_->i32_type(),
                                                 ir_->int_immediate_number(ir_->i32_type(), stmt->begin_offset),
                                                 ir_->int_immediate_number(ir_->i32_type(), 2));
        begin_expr_value = ir_->load_variable(
            ir_->struct_array_access(ir_->i32_type(), get_buffer_value(BufferType::GlobalTmps, PrimitiveType::i32),
                                     begin_idx),
            ir_->i32_type());
      } else {
        begin_expr_value = ir_->int_immediate_number(ir_->i32_type(), stmt->begin_value, false);  // Named Constant
      }
      if (!stmt->const_end) {
        spirv::Value end_idx = ir_->make_value(spv::OpShiftRightArithmetic, ir_->i32_type(),
                                               ir_->int_immediate_number(ir_->i32_type(), stmt->end_offset),
                                               ir_->int_immediate_number(ir_->i32_type(), 2));
        end_expr_value = ir_->load_variable(
            ir_->struct_array_access(ir_->i32_type(), get_buffer_value(BufferType::GlobalTmps, PrimitiveType::i32),
                                     end_idx),
            ir_->i32_type());
      } else {
        end_expr_value = ir_->int_immediate_number(ir_->i32_type(), stmt->end_value, true);
      }
    }
    total_elems = ir_->sub(end_expr_value, begin_expr_value);
    task_attribs_.advisory_total_num_threads = kMaxNumThreadsGridStrideLoop;

    // Try to extract `end_stmt` as a product of ExternalTensorShapeAlongAxisStmt so the runtime can compute the
    // actual iteration bound from the launch-time ndarray shapes and avoid dispatching `kMaxNumThreadsGridStrideLoop`
    // threads. Handles `end_stmt` itself being one shape-lookup, a BinaryOpStmt(Mul, ...) tree of shape-lookups, or
    // the same with ConstStmt(value=1) identity factors (Unit factors are common when `ndrange` has a batch dimension
    // of 1). Anything else -> leave the list empty and the advisory fallback keeps current behavior.
    if (stmt->end_stmt) {
      std::vector<TaskAttributes::RangeForAttributes::ArgShapeRef> refs;
      std::function<bool(Stmt *)> walk = [&](Stmt *s) -> bool {
        if (!s)
          return false;
        if (auto *shape = s->cast<ExternalTensorShapeAlongAxisStmt>()) {
          TaskAttributes::RangeForAttributes::ArgShapeRef ref;
          ref.arg_id = shape->arg_id;
          ref.axis = shape->axis;
          refs.push_back(std::move(ref));
          return true;
        }
        if (auto *c = s->cast<ConstStmt>()) {
          return c->val.val_int32() == 1;  // identity factor only
        }
        if (auto *binop = s->cast<BinaryOpStmt>()) {
          // Mul distributes: cap is the product of children, each of which must itself reduce to a shape.
          if (binop->op_type == BinaryOpType::mul) {
            return walk(binop->lhs) && walk(binop->rhs);
          }
          // `qd.ndrange` wraps each axis in `max(X, 1)` to guarantee a positive iteration count even when
          // the shape is zero. For capping purposes `max(X, 1)` is safe to approximate as just X: when X >= 1
          // the max is X (exact), and when X == 0 the max is 1 (we overcap by one thread, which still
          // dispatches a single-iteration kernel rather than 131072 idle threads). So if one operand is a
          // ConstStmt(value=1), walk the other and accept.
          if (binop->op_type == BinaryOpType::max) {
            // `qd.ndrange` lowers each axis to `max(0, shape)` (and sometimes `max(1, shape)` on other code
            // paths) to guarantee a non-negative iteration count - ndarray shapes are always >= 0, so
            // max(0|1, X) == X on any input that actually reaches codegen. Treat either identity as
            // pass-through so we can keep capping through the guard.
            auto *lhs_const = binop->lhs->cast<ConstStmt>();
            auto *rhs_const = binop->rhs->cast<ConstStmt>();
            auto is_max_identity = [](ConstStmt *c) {
              if (!c)
                return false;
              int v = c->val.val_int32();
              return v == 0 || v == 1;
            };
            if (is_max_identity(lhs_const))
              return walk(binop->rhs);
            if (is_max_identity(rhs_const))
              return walk(binop->lhs);
          }
          return false;
        }
        return false;
      };
      if (walk(stmt->end_stmt) && !refs.empty()) {
        range_for_attribs.end_shape_product = std::move(refs);
      }
    }
  }
  task_attribs_.advisory_num_threads_per_group = stmt->block_dim;
  ir_->debug_name(spv::OpName, begin_expr_value, "begin_expr_value");
  ir_->debug_name(spv::OpName, total_elems, total_elems_name);

  spirv::Value begin_ = ir_->add(ir_->cast(ir_->i32_type(), ir_->get_global_invocation_id(0)), begin_expr_value);
  ir_->debug_name(spv::OpName, begin_, "begin_");
  spirv::Value end_ = ir_->add(total_elems, begin_expr_value);
  ir_->debug_name(spv::OpName, end_, "end_");
  const std::string total_invocs_name = "total_invocs";
  // For now, |total_invocs_name| is equal to |total_elems|. Once we support
  // dynamic range, they will be different.
  // https://www.khronos.org/opengl/wiki/Compute_Shader#Inputs

  // HLSL & WGSL cross compilers do not support this builtin
  spirv::Value total_invocs = ir_->cast(
      ir_->i32_type(),
      ir_->mul(ir_->get_num_work_groups(0),
               ir_->uint_immediate_number(ir_->u32_type(), task_attribs_.advisory_num_threads_per_group, true)));
  /*
  const int group_x = (task_attribs_.advisory_total_num_threads +
                        task_attribs_.advisory_num_threads_per_group - 1) /
                      task_attribs_.advisory_num_threads_per_group;
  spirv::Value total_invocs = ir_->uint_immediate_number(
      ir_->i32_type(), group_x * task_attribs_.advisory_num_threads_per_group,
      false);
      */

  ir_->debug_name(spv::OpName, total_invocs, total_invocs_name);

  // Must get init label after making value(to make sure they are correct)
  spirv::Label init_label = ir_->current_label();
  spirv::Label head_label = ir_->new_label();
  spirv::Label body_label = ir_->new_label();
  spirv::Label continue_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();
  ir_->make_inst(spv::OpBranch, head_label);

  // loop head
  ir_->start_label(head_label);
  spirv::PhiValue loop_var = ir_->make_phi(begin_.stype, 2);
  ir_->register_value("ii", loop_var);
  loop_var.set_incoming(0, begin_, init_label);
  spirv::Value loop_cond = ir_->lt(loop_var, end_);
  ir_->make_inst(spv::OpLoopMerge, merge_label, continue_label, spv::LoopControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, loop_cond, body_label, merge_label);

  // loop body
  ir_->start_label(body_label);
  push_loop_control_labels(continue_label, merge_label);

  // loop kernel
  stmt->body->accept(this);
  pop_loop_control_labels();
  ir_->make_inst(spv::OpBranch, continue_label);

  // loop continue
  ir_->start_label(continue_label);
  spirv::Value next_value = ir_->add(loop_var, total_invocs);
  loop_var.set_incoming(1, next_value, ir_->current_label());
  ir_->make_inst(spv::OpBranch, head_label);

  // loop merge
  ir_->start_label(merge_label);

  emit_adstack_task_end_overflow_check();
  ir_->make_inst(spv::OpReturn);
  ir_->make_inst(spv::OpFunctionEnd);

  task_attribs_.buffer_binds = get_buffer_binds();
}

void TaskCodegen::generate_struct_for_kernel(OffloadedStmt *stmt) {
  task_attribs_.name = task_name_;
  task_attribs_.task_type = OffloadedTaskType::struct_for;
  task_attribs_.advisory_total_num_threads = 65536;
  task_attribs_.advisory_num_threads_per_group = 128;

  // The computation for a single work is wrapped inside a function, so that
  // we can do grid-strided loop.
  ir_->start_function(kernel_function_);
  ensure_any_overflow_signal_var();

  auto listgen_buffer = get_buffer_value(BufferType::ListGen, PrimitiveType::u32);
  auto listgen_count_ptr = ir_->struct_array_access(ir_->u32_type(), listgen_buffer, ir_->const_i32_zero_);
  auto listgen_count = ir_->load_variable(listgen_count_ptr, ir_->u32_type());

  auto invoc_index = ir_->get_global_invocation_id(0);

  spirv::Label loop_head = ir_->new_label();
  spirv::Label loop_body = ir_->new_label();
  spirv::Label loop_merge = ir_->new_label();

  auto loop_index_var = ir_->alloca_variable(ir_->u32_type());
  ir_->store_variable(loop_index_var, invoc_index);

  ir_->make_inst(spv::OpBranch, loop_head);
  ir_->start_label(loop_head);
  // for (; index < list_size; index += gl_NumWorkGroups.x *
  // gl_WorkGroupSize.x)
  auto loop_index = ir_->load_variable(loop_index_var, ir_->u32_type());
  auto loop_cond = ir_->make_value(spv::OpULessThan, ir_->bool_type(), loop_index, listgen_count);
  ir_->make_inst(spv::OpLoopMerge, loop_merge, loop_body, spv::LoopControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, loop_cond, loop_body, loop_merge);
  {
    ir_->start_label(loop_body);
    auto listgen_index_ptr = ir_->struct_array_access(
        ir_->u32_type(), listgen_buffer, ir_->add(ir_->uint_immediate_number(ir_->u32_type(), 1), loop_index));
    auto listgen_index = ir_->load_variable(listgen_index_ptr, ir_->u32_type());

    // kernel
    ir_->register_value("ii", listgen_index);
    stmt->body->accept(this);

    // continue
    spirv::Value total_invocs = ir_->cast(
        ir_->u32_type(),
        ir_->mul(ir_->get_num_work_groups(0),
                 ir_->uint_immediate_number(ir_->u32_type(), task_attribs_.advisory_num_threads_per_group, true)));
    auto next_index = ir_->add(loop_index, total_invocs);
    ir_->store_variable(loop_index_var, next_index);
    ir_->make_inst(spv::OpBranch, loop_head);
  }
  ir_->start_label(loop_merge);

  emit_adstack_task_end_overflow_check();
  ir_->make_inst(spv::OpReturn);       // return;
  ir_->make_inst(spv::OpFunctionEnd);  // } Close kernel

  task_attribs_.buffer_binds = get_buffer_binds();
}

// Return the address in device memory for a global/storage-buffer access.
// Only works for device-buffer-backed pointers (via ptr_to_buffers_), not
// workgroup arrays - those already have a pointer from OpAccessChain.
spirv::Value TaskCodegen::at_buffer(const Stmt *ptr, DataType dt) {
  spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

  if (ptr_val.stype.dt == PrimitiveType::u64) {
    auto elem_type = ir_->get_primitive_type(dt);
    auto ptr_elem_type = ir_->get_pointer_type(elem_type, spv::StorageClassPhysicalStorageBuffer);

    // Prefer base-pointer + OpPtrAccessChain when we have decomposed
    // components.  This makes SPIRV-Cross emit  base[index]  instead of
    // per-element  reinterpret_cast<device T*>(ulong_expr)  which
    // triggers a Metal shader compiler miscompilation bug when the stored
    // value is loop-invariant.
    auto comp_it = physical_ptr_components_.find(ptr);
    if (comp_it != physical_ptr_components_.end()) {
      auto &comp = comp_it->second;
      spirv::Value base_ptr = ir_->make_value(spv::OpConvertUToPtr, ptr_elem_type, comp.base_ptr);
      spirv::Value elem_ptr = ir_->make_value(spv::OpPtrAccessChain, ptr_elem_type, base_ptr, comp.element_index);
      elem_ptr.flag = ValueKind::kPhysicalPtr;
      return elem_ptr;
    }

    // Fallback: wrap in a single-member struct so that OpAccessChain
    // produces an lvalue.  SPIRV-Cross emits &expr for atomic operations;
    // without this wrapper the bare OpConvertUToPtr rvalue produces
    // invalid MSL on Metal.
    std::vector<std::tuple<spirv::SType, std::string, size_t>> members = {{elem_type, "_m0", 0}};
    auto wrapper_struct = ir_->create_struct_type(members);
    auto ptr_struct_type = ir_->get_pointer_type(wrapper_struct, spv::StorageClassPhysicalStorageBuffer);
    spirv::Value struct_ptr = ir_->make_value(spv::OpConvertUToPtr, ptr_struct_type, ptr_val);

    spirv::Value elem_ptr = ir_->make_value(spv::OpAccessChain, ptr_elem_type, struct_ptr, ir_->const_i32_zero_);
    elem_ptr.flag = ValueKind::kPhysicalPtr;
    return elem_ptr;
  }

  QD_ERROR_IF(!is_integral(ptr_val.stype.dt), "at_buffer failed, `ptr_val.stype.dt` is not integral. Stmt = {} : {}",
              ptr->name(), ptr->type_hint());

  spirv::Value buffer = get_buffer_value(ptr_to_buffers_.at(ptr), dt);
  size_t width = ir_->get_primitive_type_size(dt);
  spirv::Value idx_val = ir_->make_value(spv::OpShiftRightLogical, ptr_val.stype, ptr_val,
                                         ir_->uint_immediate_number(ptr_val.stype, size_t(std::log2(width))));
  spirv::Value ret = ir_->struct_array_access(ir_->get_primitive_type(dt), buffer, idx_val);
  return ret;
}

// For primitive float types, access the storage buffer through the native type view instead of
// the uint-punned view. SPIR-V / Vulkan treats each (descriptor_set, binding) as a distinct
// variable; `at_buffer` creates a new binding per (buffer, element_type) pair, so the u32 view
// of a buffer and its f32 view are different variables pointing to the same memory. Without an
// `Aliased` decoration the driver / SPIRV-Tools is free to assume they do not alias, meaning a
// plain `OpLoad` through the u32 view is not ordered against a preceding `OpAtomicFAddEXT` on
// the f32 view at the same address. The reverse-mode pattern `m.grad[i][j,k] += loss.grad;
// tmp = m.grad[i][j,k]; m.grad[i][j,k] = 0; n.grad += tmp * factor` hits this: the load reads
// the stale zero initial value, `tmp = 0`, and the adjoint never propagates (`test_ad_dynamic_index.py::
// test_matrix_non_constant_index[arch=vulkan]` asserts `0.0 == 1.0`). Using the native f32 view
// for plain load/store keeps them on the same binding as the atomic and removes the aliasing
// question entirely. Integer types (i32/u32/i16/u16/i8/u8) already route through their own uint
// view which matches the atomic path, so those stay as-is. `u1` stays on u8 because u1 has no
// native SPIR-V storage representation.
static DataType pick_buffer_access_type(DataType dt, const spirv::Value &ptr_val, spirv::IRBuilder &ir) {
  if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return PrimitiveType::u8;
  }
  if (ptr_val.stype.dt == PrimitiveType::u64) {
    return dt;
  }
  // Explicit whitelist of the real primitives we route natively, replacing the prior open-ended `is_real(dt)`
  // predicate. Any future real-like primitive (e.g. a bfloat16, or an fp8 variant) would not have an audited SPIR-V
  // storage-capability story yet - rather than silently fall into the native-view branch, it must be added here
  // deliberately after the storage-capability plumbing for its bit width is confirmed (see the
  // `CapabilityStorageBuffer{8,16}BitAccess` emissions in `spirv_ir_builder.cpp`).
  if (dt->is_primitive(PrimitiveTypeID::f16) || dt->is_primitive(PrimitiveTypeID::f32) ||
      dt->is_primitive(PrimitiveTypeID::f64)) {
    return dt;
  }
  return ir.get_quadrants_uint_type(dt);
}

spirv::Value TaskCodegen::load_buffer(const Stmt *ptr, DataType dt, bool is_volatile) {
  spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

  DataType ti_buffer_type = pick_buffer_access_type(dt, ptr_val, *ir_);

  auto buf_ptr = at_buffer(ptr, ti_buffer_type);
  // The Metal-global `use_volatile_buffer_access_` flag (set on every buffer in the constructor) marks the storage
  // *buffer* as volatile, which protects against MoltenVK's coarse-grained LICM bug.  `is_volatile` here is a
  // per-load opt-in for `qd.volatile_load`: the OpLoad itself carries the `Volatile` `MemoryAccess` mask so the
  // SPIR-V optimiser cannot forward / merge this specific read with prior reads of the same address, even when
  // the surrounding buffer is not blanket-decorated.
  auto val_bits = is_volatile ? ir_->load_variable_volatile(buf_ptr, ir_->get_primitive_type(ti_buffer_type))
                              : ir_->load_variable(buf_ptr, ir_->get_primitive_type(ti_buffer_type));
  if (dt->is_primitive(PrimitiveTypeID::u1))
    return ir_->cast(ir_->bool_type(), val_bits);
  return ti_buffer_type == dt ? val_bits : ir_->make_value(spv::OpBitcast, ir_->get_primitive_type(dt), val_bits);
}

void TaskCodegen::store_buffer(const Stmt *ptr, spirv::Value val) {
  spirv::Value ptr_val = ir_->query_value(ptr->raw_name());

  DataType ti_buffer_type = pick_buffer_access_type(val.stype.dt, ptr_val, *ir_);
  if (val.stype.dt->is_primitive(PrimitiveTypeID::u1)) {
    // Stores go through i8 (matching the original path) so a signed i1 narrowing is preserved.
    ti_buffer_type = PrimitiveType::i8;
  }

  auto buf_ptr = at_buffer(ptr, ti_buffer_type);
  spirv::Value val_bits;
  if (val.stype.dt == ti_buffer_type) {
    val_bits = val;
  } else if (val.stype.dt->is_primitive(PrimitiveTypeID::u1)) {
    // SPIR-V `OpBitcast` rejects bool operands (spec: operand must be numerical scalar / vector or pointer). A direct
    // `OpBitcast %char %bool_val` for a `u1` field / ndarray store would validate as `Expected input to be a pointer or
    // int or float vector or scalar: Bitcast`; most drivers ignore that and crash inside the pipeline compiler
    // (observed on Mesa RADV: a hard SIGSEGV inside `libvulkan_radeon.so::create_compute_pipeline` the moment the
    // offending kernel is registered). Route through `IRBuilder::cast`, which lowers `bool -> int` to `OpSelect`
    // picking `1` or `0` of the target type - the canonical spec-compliant way to widen a bool, matching what
    // `load_buffer` already does on the reverse path and keeping the "bool serialises as 0 / 1" behaviour every user of
    // `to_numpy()` / `from_numpy()` depends on.
    val_bits = ir_->cast(ir_->get_primitive_type(ti_buffer_type), val);
  } else {
    val_bits = ir_->make_value(spv::OpBitcast, ir_->get_primitive_type(ti_buffer_type), val);
  }
  ir_->store_variable(buf_ptr, val_bits);
}

spirv::Value TaskCodegen::get_buffer_value(BufferInfo buffer, DataType dt) {
  auto type = ir_->get_primitive_type(dt);
  auto key = std::make_pair(buffer, type.id);

  const auto it = buffer_value_map_.find(key);
  if (it != buffer_value_map_.end()) {
    return it->second;
  }

  if (buffer.type == BufferType::Args) {
    compile_args_struct();

    buffer_binding_map_[key] = 0;
    buffer_value_map_[key] = args_buffer_value_;
    return args_buffer_value_;
  }

  if (buffer.type == BufferType::Rets) {
    compile_ret_struct();

    buffer_binding_map_[key] = 1;
    buffer_value_map_[key] = ret_buffer_value_;
    return ret_buffer_value_;
  }

  // Binding head starts at 2, so we don't break args and rets
  int binding = binding_head_++;
  buffer_binding_map_[key] = binding;

  spirv::Value buffer_value = ir_->buffer_argument(type, 0, binding, buffer_instance_name(buffer));
  if (use_volatile_buffer_access_) {
    ir_->decorate(spv::OpDecorate, buffer_value, spv::DecorationVolatile);
  }
  buffer_value_map_[key] = buffer_value;

  // Type-punned views of the same underlying `VkBuffer` need an explicit `Aliased` decoration. Every
  // `(BufferInfo, element_type)` pair here gets its own `OpVariable` with a fresh `DescriptorSet` /
  // `Binding`, so a field read through the `u32` view and a preceding `OpAtomicFAddEXT` through the
  // `f32` view are separate SPIR-V variables pointing to the same memory. Without `Aliased` the
  // driver is spec-free to assume they don't alias, and the plain load is not ordered against the
  // atomic write -- the load reads the stale zero initial value and the reverse-mode adjoint silently
  // drops to zero (`test_ad_dynamic_index.py::test_matrix_non_constant_index[arch=vulkan]` reproduces
  // this on every device that exposes `shaderBufferFloat32AtomicAdd`).
  //
  // The aliasing hazard applies across every pairing of views on the same buffer, not just
  // (native-float atomic, plain load): the CAS-emulation atomic path on devices without
  // `shaderBufferFloat32AtomicAdd` routes float atomics through the `u32` view while other sites may
  // emit atomics min / max / mul directly against the `u32` view -- each such pairing against a
  // plain-load-through-any-view needs the same decoration. Decorating here rather than at first
  // `at_buffer` call keeps the fix independent of which call site happens to introduce the second
  // view.
  //
  // Decorate lazily: a single-view buffer stays un-decorated (no perf cost from disabled
  // cross-variable scheduling optimizations), and only when a buffer gets its second distinct view do
  // we retroactively decorate every view -- existing ones through the id-set guard below, and the new
  // one unconditionally in the same sweep.
  auto &views = buffer_views_by_buffer_[buffer];
  views.push_back(buffer_value);
  if (views.size() >= 2) {
    for (const auto &v : views) {
      if (aliased_decorated_buffer_ids_.insert(v.id).second) {
        ir_->decorate(spv::OpDecorate, v, spv::DecorationAliased);
      }
    }
  }
  QD_TRACE("buffer name = {}, value = {}", buffer_instance_name(buffer), buffer_value.id);

  return buffer_value;
}

spirv::Value TaskCodegen::make_pointer(size_t offset) {
  if (use_64bit_pointers) {
    // This is hacky, should check out how to encode uint64 values in spirv
    return ir_->uint_immediate_number(ir_->u64_type(), offset);
  } else {
    return ir_->uint_immediate_number(ir_->u32_type(), uint32_t(offset));
  }
}

void TaskCodegen::compile_args_struct() {
  if (!ctx_attribs_->has_args())
    return;

  // Generate struct IR
  tinyir::Block blk;
  std::unordered_map<std::vector<int>, const tinyir::Type *, hashing::Hasher<std::vector<int>>> element_types;
  std::unordered_map<std::vector<int>, const quadrants::lang::Type *, hashing::Hasher<std::vector<int>>>
      element_quadrants_types;
  std::vector<const tinyir::Type *> root_element_types;
  bool has_buffer_ptr = caps_->get(DeviceCapability::spirv_has_physical_storage_buffer);
  std::function<void(const std::vector<int> &indices, const Type *type)> add_types_to_element_types =
      [&](const std::vector<int> &indices, const Type *type) {
        auto spirv_type = translate_ti_type(blk, type, has_buffer_ptr);
        if (auto struct_type = type->cast<quadrants::lang::StructType>()) {
          for (int j = 0; j < struct_type->elements().size(); ++j) {
            std::vector<int> indices_copy = indices;
            indices_copy.push_back(j);
            add_types_to_element_types(indices_copy, struct_type->elements()[j].type);
          }
        }
        element_quadrants_types[indices] = type;
        element_types[indices] = spirv_type;
      };
  for (int i = 0; i < ctx_attribs_->args_type()->elements().size(); i++) {
    auto *type = ctx_attribs_->args_type()->elements()[i].type;
    auto spirv_type = translate_ti_type(blk, type, has_buffer_ptr);
    element_types[{i}] = spirv_type;
    element_quadrants_types[{i}] = type;
    root_element_types.push_back(spirv_type);
    if (auto struct_type = type->cast<quadrants::lang::StructType>()) {
      for (int j = 0; j < struct_type->elements().size(); ++j) {
        add_types_to_element_types({i, j}, struct_type->elements()[j].type);
      }
    }
  }
  const tinyir::Type *struct_type = blk.emplace_back<StructType>(root_element_types);

  // Reduce struct IR
  std::unordered_map<const tinyir::Type *, const tinyir::Type *> old2new;
  auto reduced_blk = ir_reduce_types(&blk, old2new);
  struct_type = old2new[struct_type];

  for (auto &element : root_element_types) {
    element = old2new[element];
  }
  for (auto &element : element_types) {
    element.second = old2new[element.second];
  }

  // Layout & translate to SPIR-V
  STD140LayoutContext layout_ctx;
  auto ir2spirv_map = ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
  args_struct_type_.id = ir2spirv_map[struct_type];

  // Must use the same type in ArgLoadStmt as in the args struct,
  // otherwise the validation will fail.
  for (auto &element : element_types) {
    spirv::SType spirv_type;
    spirv_type.id = ir2spirv_map.at(element.second);
    spirv_type.dt = element_quadrants_types[element.first];
    args_struct_types_[element.first] = spirv_type;
  }

  args_buffer_value_ = ir_->uniform_struct_argument(args_struct_type_, 0, 0, "args");
}

void TaskCodegen::compile_ret_struct() {
  if (!ctx_attribs_->has_rets())
    return;

  // Generate struct IR
  tinyir::Block blk;
  std::vector<const tinyir::Type *> element_types;
  bool has_buffer_ptr = caps_->get(DeviceCapability::spirv_has_physical_storage_buffer);
  for (auto &element : ctx_attribs_->rets_type()->elements()) {
    element_types.push_back(translate_ti_type(blk, element.type, has_buffer_ptr));
  }
  const tinyir::Type *struct_type = blk.emplace_back<StructType>(element_types);

  // Reduce struct IR
  std::unordered_map<const tinyir::Type *, const tinyir::Type *> old2new;
  auto reduced_blk = ir_reduce_types(&blk, old2new);
  struct_type = old2new[struct_type];

  for (auto &element : element_types) {
    element = old2new[element];
  }

  // Layout & translate to SPIR-V
  STD430LayoutContext layout_ctx;
  auto ir2spirv_map = ir_translate_to_spirv(reduced_blk.get(), layout_ctx, ir_.get());
  ret_struct_type_.id = ir2spirv_map[struct_type];

  rets_struct_types_.resize(element_types.size());
  for (int i = 0; i < element_types.size(); i++) {
    rets_struct_types_[i].id = ir2spirv_map.at(element_types[i]);
    if (i < ctx_attribs_->rets_type()->elements().size()) {
      rets_struct_types_[i].dt = ctx_attribs_->rets_type()->get_element_type(std::array{i});
    } else {
      rets_struct_types_[i].dt = PrimitiveType::i32;
    }
  }

  ret_buffer_value_ = ir_->buffer_struct_argument(ret_struct_type_, 0, 1, "rets");
}

std::vector<BufferBind> TaskCodegen::get_buffer_binds() {
  std::vector<BufferBind> result;
  for (auto &[key, val] : buffer_binding_map_) {
    result.push_back(BufferBind{key.first, int(val)});
  }
  return result;
}

// --- AdStack (autodiff local-variable history stack) for SPIR-V ---
// Primal and adjoint history lives in two per-dispatch StorageBuffers: one of Array<f32> bound as
// BufferType::AdStackHeapFloat (for f32-valued adstacks) and one of Array<i32> bound as BufferType::AdStackHeapInt
// (for i32 and u1 adstacks; u1 is stored as i32 because OpTypeBool has no defined storage layout and
// `get_array_type` already applies the same bool->int remap on the Function-scope path). Each SPIR-V invocation
// owns a contiguous slice sized at `ad_stack_heap_per_thread_stride_*` elements of the respective type; within
// that slice, each adstack holds `max_size` primals followed by `max_size` adjoints. The runtime reads
// `task_attribs.ad_stack_heap_per_thread_stride_(float|int)` and multiplies by the dispatched invocation count
// to size each buffer at launch. `count_var` stays in Function-scope because it is just one u32 per adstack and
// is hot on every push/pop, so keeping it on-chip avoids a device-memory round-trip per push.
//
// Heap storage is required on Metal/MoltenVK because the per-thread shader private-memory budget (a few dozen
// KB per invocation) is too small for deeply nested reverse-mode kernels. A kernel that declares on the order
// of 100+ u1/i32 adstack slots at default capacity 256 easily exceeds that budget (e.g. ~130 KB per thread),
// so all such adstacks must live in device memory, not on-chip.
//
// Other primitive types (f64, i64, ...) are rejected outright by `visit(AdStackAllocaStmt)`. Add dedicated
// f64/i64 heap buffers if support for those element types becomes necessary.

spirv::Value TaskCodegen::get_ad_stack_heap_buffer_float() {
  if (ad_stack_heap_buffer_float_.id == 0) {
    ad_stack_heap_buffer_float_ = get_buffer_value({BufferType::AdStackHeapFloat}, PrimitiveType::f32);
  }
  return ad_stack_heap_buffer_float_;
}

spirv::Value TaskCodegen::get_ad_stack_metadata_buffer() {
  if (ad_stack_metadata_buffer_.id == 0) {
    ad_stack_metadata_buffer_ = get_buffer_value({BufferType::AdStackMetadata}, PrimitiveType::u32);
  }
  return ad_stack_metadata_buffer_;
}

spirv::Value TaskCodegen::get_ad_stack_metadata_stride_float() {
  if (ad_stack_metadata_stride_float_.id == 0) {
    spirv::Value buf = get_ad_stack_metadata_buffer();
    spirv::Value ptr = ir_->struct_array_access(ir_->u32_type(), buf, ir_->uint_immediate_number(ir_->i32_type(), 0));
    ad_stack_metadata_stride_float_ = ir_->load_variable(ptr, ir_->u32_type());
  }
  return ad_stack_metadata_stride_float_;
}

spirv::Value TaskCodegen::get_ad_stack_metadata_stride_int() {
  if (ad_stack_metadata_stride_int_.id == 0) {
    spirv::Value buf = get_ad_stack_metadata_buffer();
    spirv::Value ptr = ir_->struct_array_access(ir_->u32_type(), buf, ir_->uint_immediate_number(ir_->i32_type(), 1));
    ad_stack_metadata_stride_int_ = ir_->load_variable(ptr, ir_->u32_type());
  }
  return ad_stack_metadata_stride_int_;
}

spirv::Value TaskCodegen::get_ad_stack_heap_thread_base_float() {
  // `row_id * per_thread_stride`. `row_id` is loaded fresh at every call from the Function-scope
  // `ad_stack_row_id_var_float_` (declared at the first alloca visit, written at the float Lowest Common Ancestor (LCA)
  // block claim site), and the resulting OpIMul lives in the call-site's basic block. Re-emitting per call site (rather
  // than caching one `row_id * stride` SSA at the alloca site and reusing it at every push / load-top) is mandatory:
  // `row_id` is a Function-scope variable load, so every load yields a fresh SSA whose definition lives in the loading
  // block; reusing one SSA across sibling blocks of the LCA would violate SPIR-V section 2.16 dominance. The cost is
  // cheap (one OpLoad + one OpIMul per push / load-top) and spirv-opt / spirv-cross can still hoist or CSE redundant
  // loads within a single basic block. Widened to u64 when the device has Int64 because `row_id * stride` can wrap u32
  // on deeply-allocated kernels and a silent wrap aliases threads into one another's heap slice.
  spirv::Value row_id = ir_->load_variable(ad_stack_row_id_var_float_, ir_->u32_type());
  spirv::Value stride_u32 = get_ad_stack_metadata_stride_float();
  if (caps_->get(DeviceCapability::spirv_has_int64)) {
    // `make_value(OpUConvert, ...)` directly rather than `ir_->cast()`: `cast()` between two unsigned integer types of
    // different widths emits `OpUConvert` followed by `OpBitcast` to `dst_type`, and with widening u32->u64 both sides
    // are already unsigned, so the trailing `OpBitcast(u64, u64)` has identical operand and result types - which SPIR-V
    // section 3.42.16 forbids; `spirv-val` rejects the shader and MoltenVK may silently refuse to compile it.
    spirv::Value row_id_u64 = ir_->make_value(spv::OpUConvert, ir_->u64_type(), row_id);
    spirv::Value stride_u64 = ir_->make_value(spv::OpUConvert, ir_->u64_type(), stride_u32);
    return ir_->mul(row_id_u64, stride_u64);
  }
  return ir_->mul(row_id, stride_u32);
}

spirv::Value TaskCodegen::ad_stack_heap_float_ptr(spirv::Value slot_offset, spirv::Value count) {
  spirv::Value base = get_ad_stack_heap_thread_base_float();
  spirv::SType idx_type = caps_->get(DeviceCapability::spirv_has_int64) ? ir_->u64_type() : ir_->u32_type();
  // `slot_offset` is a u32 load from the metadata buffer; widen it to the index type alongside `count`. See
  // `get_ad_stack_heap_thread_base_float` for why we widen via `OpUConvert` directly.
  spirv::Value offset_idx = caps_->get(DeviceCapability::spirv_has_int64)
                                ? ir_->make_value(spv::OpUConvert, idx_type, slot_offset)
                                : slot_offset;
  spirv::Value count_idx =
      caps_->get(DeviceCapability::spirv_has_int64) ? ir_->make_value(spv::OpUConvert, idx_type, count) : count;
  spirv::Value heap_index = ir_->add(ir_->add(base, offset_idx), count_idx);
  return ir_->struct_array_access(ir_->f32_type(), get_ad_stack_heap_buffer_float(), heap_index);
}

spirv::Value TaskCodegen::get_ad_stack_heap_buffer_int() {
  if (ad_stack_heap_buffer_int_.id == 0) {
    ad_stack_heap_buffer_int_ = get_buffer_value({BufferType::AdStackHeapInt}, PrimitiveType::i32);
  }
  return ad_stack_heap_buffer_int_;
}

spirv::Value TaskCodegen::get_ad_stack_heap_thread_base_int() {
  // Eager `gl_GlobalInvocationID * stride_int` per-thread layout. The int heap backs loop-index recovery and if-branch
  // flag adstacks, which the autodiff pass emits unconditionally at the offload body root for reverse-pass control-flow
  // replay; folding those root-level pushes into the float lazy-row-claim Lowest Common Ancestor (LCA) block
  // computation would pull the LCA up to the offload root and eliminate the float-heap savings. Per-thread layout is
  // correctness-equivalent to the prior single-counter mechanism for the int heap and keeps the heap allocation
  // trivially predictable at `dispatched_threads * stride_int * sizeof(i32)` - small enough not to matter (per-thread
  // int strides typically stay in the tens of i32 entries, two orders of magnitude below the float strides whose
  // worst-case footprint motivated this change). The same u64 widening rule applies for the same wrap-aliasing reason
  // as the float counterpart.
  spirv::Value row_id = ir_->get_global_invocation_id(0);
  spirv::Value stride_u32 = get_ad_stack_metadata_stride_int();
  if (caps_->get(DeviceCapability::spirv_has_int64)) {
    spirv::Value row_id_u64 = ir_->make_value(spv::OpUConvert, ir_->u64_type(), row_id);
    spirv::Value stride_u64 = ir_->make_value(spv::OpUConvert, ir_->u64_type(), stride_u32);
    return ir_->mul(row_id_u64, stride_u64);
  }
  return ir_->mul(row_id, stride_u32);
}

spirv::Value TaskCodegen::ad_stack_heap_int_ptr(spirv::Value slot_offset, spirv::Value count) {
  spirv::Value base = get_ad_stack_heap_thread_base_int();
  spirv::SType idx_type = caps_->get(DeviceCapability::spirv_has_int64) ? ir_->u64_type() : ir_->u32_type();
  spirv::Value offset_idx = caps_->get(DeviceCapability::spirv_has_int64)
                                ? ir_->make_value(spv::OpUConvert, idx_type, slot_offset)
                                : slot_offset;
  spirv::Value count_idx =
      caps_->get(DeviceCapability::spirv_has_int64) ? ir_->make_value(spv::OpUConvert, idx_type, count) : count;
  spirv::Value heap_index = ir_->add(ir_->add(base, offset_idx), count_idx);
  return ir_->struct_array_access(ir_->i32_type(), get_ad_stack_heap_buffer_int(), heap_index);
}

// Lazily load the per-alloca `(offset, max_size)` pair for `info` from the AdStackMetadata buffer and cache the
// SSA ids on `info`. The metadata buffer layout is `[stride_float, stride_int, offset_0, max_size_0, offset_1,
// max_size_1, ...]` so slot i lives at buffer indices `2 + 2*i` and `2 + 2*i + 1`. First-call emission happens
// at the first push / load-top / load-top-adj site for this alloca (the AllocaStmt visitor sets `stack_id` and
// caches the buffer + stride eagerly so it dominates every sibling body). The adjoint offset for f32 adstacks
// is a derived `OpIAdd` rather than an extra buffer load - mirrors the host launcher's `offset + max_size`
// prefix-sum layout for the primal/adjoint pair.
// Allocate the per-task overflow-signal accumulator at the function entry block and zero-initialize it.
// Holds the maximum `stack_id + 1` value seen across all overflowing push sites in this thread; 0 means no
// overflow. Read + conditionally atomic-max'd to the host-visible AdStackOverflow buffer at task-end.
// Called eagerly from each `generate_*_kernel` at task start so the init-store dominates every push site
// and every task-end read; lazy alloc + init at the first push site would put the store inside whatever
// conditional block the first push happened to live in, leaving the var undefined on paths that bypass it.
spirv::Value TaskCodegen::ensure_any_overflow_signal_var() {
  if (any_overflow_signal_var_.id != 0) {
    return any_overflow_signal_var_;
  }
  any_overflow_signal_var_ = ir_->alloca_variable(ir_->u32_type());
  ir_->store_variable(any_overflow_signal_var_, ir_->uint_immediate_number(ir_->u32_type(), 0));
  return any_overflow_signal_var_;
}

// Emit the task-end overflow check at the current insertion point, just before the kernel's `OpReturn`.
// Reads `any_overflow_signal_var_`; if non-zero, atomic-max'es it into slot 0 of the host-visible
// AdStackOverflow buffer. The buffer is host-readable (Apple Silicon shared memory; Vulkan
// HOST_VISIBLE | HOST_COHERENT), so the host polls slot 0 directly without any DtoH or sync drain. No-op
// when the task has no adstack push sites (the var is unallocated).
void TaskCodegen::emit_adstack_task_end_overflow_check() {
  // Skip the entire emit (including the AdStackOverflow / AdStackTaskRegistryId buffer accesses) when
  // the task body never visited an `AdStackPushStmt`. Forward-only tasks would otherwise force the
  // launcher's bind path to wire AdStackTaskRegistryId for kernels where
  // `publish_adstack_metadata_spirv` never allocated the buffer (no task in the kernel has adstacks),
  // crashing Metal's `rw_buffer` device-equality assertion on the kDeviceNullAllocation fallback.
  if (!task_has_adstack_push_ || any_overflow_signal_var_.id == 0) {
    return;
  }
  spirv::Value zero = ir_->uint_immediate_number(ir_->u32_type(), 0);
  spirv::Value cur = ir_->load_variable(any_overflow_signal_var_, ir_->u32_type());
  spirv::Value has_overflow = ir_->ne(cur, zero);
  spirv::Label then_label = ir_->new_label();
  spirv::Label merge_label = ir_->new_label();
  ir_->make_inst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
  ir_->make_inst(spv::OpBranchConditional, has_overflow, then_label, merge_label);
  ir_->start_label(then_label);
  {
    spirv::Value overflow_buffer = get_buffer_value(BufferType::AdStackOverflow, PrimitiveType::u32);
    spirv::Value overflow_signal_ptr =
        ir_->struct_array_access(ir_->u32_type(), overflow_buffer, ir_->uint_immediate_number(ir_->i32_type(), 0));
    ir_->make_value(spv::OpAtomicUMax, ir_->u32_type(), overflow_signal_ptr,
                    /*scope=*/ir_->const_i32_one_,
                    /*semantics=*/ir_->const_i32_zero_, cur);
    // Record the offending task's `Program::adstack_sizing_info_registry_` id into slot 1 via
    // `cmpxchg(0, registry_id)`. The launcher pre-writes the registry id into
    // `AdStackTaskRegistryId[task_id_in_kernel]` per task; the codegen reads that slot at the
    // task-end emit and atomically swaps it into AdStackOverflow[1] when the latter is still 0
    // (i.e. this is the FIRST overflowing thread across all tasks in the dispatch). The host
    // raise site reads slot 1 and routes through `Program::diagnose_adstack_overflow_message` to
    // produce a kernel-name + offload-task-index identity block.
    spirv::Value task_registry_buffer = get_buffer_value(BufferType::AdStackTaskRegistryId, PrimitiveType::u32);
    spirv::Value task_registry_ptr = ir_->struct_array_access(
        ir_->u32_type(), task_registry_buffer, ir_->uint_immediate_number(ir_->i32_type(), task_id_in_kernel_));
    spirv::Value registry_id = ir_->load_variable(task_registry_ptr, ir_->u32_type());
    spirv::Value overflow_task_id_ptr =
        ir_->struct_array_access(ir_->u32_type(), overflow_buffer, ir_->uint_immediate_number(ir_->i32_type(), 1));
    ir_->make_value(spv::OpAtomicCompareExchange, ir_->u32_type(), overflow_task_id_ptr,
                    /*scope=*/ir_->const_i32_one_, /*sem_eq=*/ir_->const_i32_zero_,
                    /*sem_neq=*/ir_->const_i32_zero_, registry_id, /*comparator=*/zero);
    ir_->make_inst(spv::OpBranch, merge_label);
  }
  ir_->start_label(merge_label);
}

void TaskCodegen::ensure_ad_stack_metadata_loaded(AdStackSpirv &info) {
  if (info.offset_val.id != 0) {
    return;
  }
  spirv::Value buf = get_ad_stack_metadata_buffer();
  uint32_t header = 2;  // slots 0, 1 are the two strides
  spirv::Value off_idx = ir_->uint_immediate_number(ir_->i32_type(), header + 2u * info.stack_id);
  spirv::Value max_idx = ir_->uint_immediate_number(ir_->i32_type(), header + 2u * info.stack_id + 1);
  spirv::Value off_ptr = ir_->struct_array_access(ir_->u32_type(), buf, off_idx);
  spirv::Value max_ptr = ir_->struct_array_access(ir_->u32_type(), buf, max_idx);
  info.offset_val = ir_->load_variable(off_ptr, ir_->u32_type());
  info.max_size_val = ir_->load_variable(max_ptr, ir_->u32_type());
  if (info.heap_kind == AdStackHeapKind::heap_float) {
    info.adjoint_offset_val = ir_->add(info.offset_val, info.max_size_val);
  }
}

void TaskCodegen::visit(AdStackAllocaStmt *stmt) {
  // `stmt->max_size == 0` is the sentinel `determine_ad_stack_size` leaves on allocas whose bound did not fold
  // to a compile-time constant but has a captured symbolic `size_expr`; the host launcher evaluates the expr at
  // each dispatch and publishes the runtime bound into the `AdStackMetadata` buffer. If both `max_size == 0`
  // and `size_expr` is null, the pre-pass should already have raised `QD_ERROR` - so reaching codegen with that
  // combination is a pass-ordering bug.
  QD_ASSERT_INFO(stmt->max_size > 0 || stmt->size_expr,
                 "Adaptive autodiff stack's size should have been determined or at least have a captured "
                 "SizeExpr by codegen time.");

  AdStackSpirv info;
  info.elem_type = ir_->get_primitive_type(stmt->ret_type);
  info.max_size_compile_time = uint32_t(stmt->max_size);
  info.stack_id = uint32_t(task_attribs_.ad_stack.allocas.size());
  TaskAttributes::AdStackAllocaAttribs attribs;
  attribs.max_size_compile_time = uint32_t(stmt->max_size);
  // Serialise the per-alloca `SizeExpr` captured by the `determine_ad_stack_size` pre-pass so the host launcher
  // can evaluate it against the live field state before each dispatch. An empty `size_expr` (null `size_expr`
  // on the stmt, typical for Bellman-Ford-resolved const bounds and for the offline-cache-hit path before
  // serialisation lands symbolic trees) means "use `max_size_compile_time`"; the runtime checks
  // `size_expr.nodes.empty()` to decide.
  if (stmt->size_expr) {
    attribs.size_expr = stmt->size_expr->serialize();
  }
  // f32 adstacks go on the f32 heap; i32 and u1 adstacks share the int heap. 64-bit primitives (f64, i64, u64)
  // are deliberately rejected here rather than falling back to Function-scope: the Function-scope path has been
  // shown unusable for real reverse-mode kernels (private-memory blowup on Metal/MoltenVK) so silently taking it
  // would paper over a correctness/perf cliff. If support is ever needed, add dedicated f64/i64 heap buffers
  // alongside the float/int ones - not a Function-scope fallback.
  if (stmt->ret_type == PrimitiveType::f32) {
    info.heap_kind = AdStackHeapKind::heap_float;
    info.offset_in_elems_compile_time = ad_stack_heap_next_offset_float_;
    ad_stack_heap_next_offset_float_ += 2u * uint32_t(stmt->max_size);
    attribs.heap_kind = TaskAttributes::AdStackAllocaAttribs::HeapKind::Float;
    attribs.offset_in_elems_compile_time = info.offset_in_elems_compile_time;
  } else if (stmt->ret_type == PrimitiveType::i32 || stmt->ret_type == PrimitiveType::u1) {
    info.heap_kind = AdStackHeapKind::heap_int;
    info.offset_in_elems_compile_time = ad_stack_heap_next_offset_int_;
    ad_stack_heap_next_offset_int_ += uint32_t(stmt->max_size);
    attribs.heap_kind = TaskAttributes::AdStackAllocaAttribs::HeapKind::Int;
    attribs.offset_in_elems_compile_time = info.offset_in_elems_compile_time;
  } else {
    QD_ERROR(
        "Reverse-mode AD on the SPIR-V backend supports only f32, i32, and u1 loop-carried variables. Got {} - "
        "cast to qd.f32 or qd.i32 in the differentiable section.",
        stmt->ret_type.to_string());
  }
  // The count slot OpAccessChain and its zero-init are emitted only after the type check above has accepted the
  // alloca. Calling `ad_stack_count_ptr` for an unsupported type would trip its `num_ad_stacks_ > 0` assert because
  // the pre-pass scan only counts the f32 / i32 / u1 cases (matching this same dispatch).
  info.count_var = ad_stack_count_ptr(info.stack_id);
  ir_->store_variable(info.count_var, ir_->uint_immediate_number(ir_->u32_type(), 0));
  // Load `(offset_val, max_size_val)` from the AdStackMetadata buffer eagerly at the alloca site so the OpLoads land
  // in the alloca enclosing block. SPIR-V section 2.16 requires every SSA definition to dominate all its uses, and
  // push/load-top/acc-adjoint sites can live in sibling blocks (forward loop body vs. backward loop body) that
  // neither dominates the other. Loading lazily at the first push site would cache SSA ids defined in the forward
  // body and reuse them from the backward body, which `spirv-val` rejects and strict drivers (MoltenVK, Adreno)
  // refuse at pipeline creation. Eager emission mirrors the existing discipline for
  // `get_ad_stack_heap_thread_base_{float,int}()` at the same site.
  ensure_ad_stack_metadata_loaded(info);
  task_attribs_.ad_stack.allocas.push_back(std::move(attribs));
  ad_stacks_[stmt] = info;
}

spirv::Value TaskCodegen::ad_stack_count_ptr(uint32_t stack_id) {
  // First call lazily allocates a single Function-scope `uint[num_ad_stacks_]` array shared across every adstack.
  // Each push / pop / load-top accesses its slot via OpAccessChain on this array - critically, an OpAccessChain
  // into a Function-scope array element is NOT promoted by spirv-opt's `LocalMultiStoreElim` / `SSARewrite` to
  // per-element phi nodes the way a per-stack scalar OpVariable is. Without this sharing, hundreds of per-stack
  // `count_var` OpVariables flowing through any enclosing loop become hundreds of `OpPhi` entries at the loop
  // header, which spirv-cross emits as one `uint _N;` forward-decl plus one `_N = _N;` alias copy per predecessor
  // branch in MSL - the dominant size amplifier on reverse-grad kernels with many adstacks.
  if (ad_stack_count_array_var_.id == 0) {
    QD_ASSERT(num_ad_stacks_ > 0);
    spirv::SType arr_type = ir_->get_function_array_type(ir_->u32_type(), num_ad_stacks_);
    ad_stack_count_array_var_ = ir_->alloca_variable(arr_type);
  }
  spirv::SType ptr_type = ir_->get_pointer_type(ir_->u32_type(), spv::StorageClassFunction);
  spirv::Value idx_const = ir_->uint_immediate_number(ir_->i32_type(), stack_id);
  return ir_->make_value(spv::OpAccessChain, ptr_type, ad_stack_count_array_var_, idx_const);
}

// Resolve the primal- or adjoint-slot pointer for `info` at index `idx`. The returned pointer is typed after the
// adstack's backing storage (f32 for heap_float, i32 for heap_int) - which is the same as `info.elem_type` except
// for u1 adstacks on the int heap, where backing is i32. Callers must explicitly convert between u1 and i32 via
// `ir_->cast(...)` at store/load sites in that single special case. `primal=true` selects the primal half of the
// slice, `false` selects the adjoint half. The per-alloca `offset` and (for f32 only) `adjoint_offset = offset +
// max_size` come from the `AdStackMetadata` buffer via `ensure_ad_stack_metadata_loaded`; int-heap adstacks have
// no adjoint slice and hit `QD_ASSERT(primal)` in that path.
spirv::Value TaskCodegen::ad_stack_slot_ptr(AdStackSpirv &info, spirv::Value idx, bool primal) {
  ensure_ad_stack_metadata_loaded(info);
  spirv::Value slot_offset = primal ? info.offset_val : info.adjoint_offset_val;
  if (info.heap_kind == AdStackHeapKind::heap_float) {
    return ad_stack_heap_float_ptr(slot_offset, idx);
  }
  QD_ASSERT(info.heap_kind == AdStackHeapKind::heap_int);
  QD_ASSERT_INFO(primal, "int-heap adstacks have no adjoint slot; auto_diff's is_real guard should have suppressed");
  return ad_stack_heap_int_ptr(slot_offset, idx);
}

// Returns the SType used for the underlying store/load on `info`'s backing storage. Matches `info.elem_type`
// except for u1 on the int heap, which is stored as i32 (and converted at the push/load sites).
spirv::SType TaskCodegen::ad_stack_backing_type(const AdStackSpirv &info) const {
  if (info.heap_kind == AdStackHeapKind::heap_int && info.elem_type.dt->is_primitive(PrimitiveTypeID::u1)) {
    return ir_->i32_type();
  }
  return info.elem_type;
}

void TaskCodegen::visit(AdStackPushStmt *stmt) {
  task_has_adstack_push_ = true;
  auto &info = ad_stacks_.at(stmt->stack);
  ensure_ad_stack_metadata_loaded(info);
  spirv::Value count = ir_->load_variable(info.count_var, ir_->u32_type());
  spirv::SType backing_type = ad_stack_backing_type(info);
  spirv::Value val = ir_->query_value(stmt->v->raw_name());
  if (info.elem_type.id != backing_type.id) {
    val = ir_->cast(backing_type, val);  // u1 -> i32 for the heap_int path
  }
  spirv::Value one = ir_->uint_immediate_number(ir_->u32_type(), 1);

  // Autodiff-bootstrap const-init pushes on the float heap: keep `count_var` balanced with the matching reverse pop,
  // but skip the slot store. These pushes execute on every thread regardless of any later gating, while the float heap
  // row claim only fires on threads that reach the LCA (inside the gate); skipping the LCA contribution (handled in the
  // pre-pass above) is what shrinks the heap, but it leaves `row_id_var` as UINT32_MAX for never-gated threads, so a
  // slot store here would write the bootstrap value into row UINT32_MAX (out of bounds, arbitrary heap corruption).
  // Dropping the store is safe because the matching reverse pop never reads the slot back via `load_top` - it only
  // mutates `count_var`. Limited to the pre-pass-recognized bootstrap set so non-bootstrap const pushes (e.g.
  // const-folded payloads at deeper sites) keep their slot stores.
  if (info.heap_kind != AdStackHeapKind::heap_int && ad_stack_bootstrap_pushes_.count(stmt) != 0) {
    ir_->store_variable(info.count_var, ir_->add(count, one));
    return;
  }

  // Plain store + count increment. Always-on overflow detection runs alongside the store via a
  // thread-local OpUMax accumulator (`any_overflow_signal_var_`); the per-push cost is two register
  // operations (compare + max-update). One conditional `OpAtomicUMax` to the host-visible AdStackOverflow
  // buffer is emitted ONCE per task at task-end (see `emit_adstack_task_end_overflow_check`), so push
  // sites do not touch global memory for overflow signaling.
  spirv::Value max_val = info.max_size_val;
  spirv::Value primal_ptr = ad_stack_slot_ptr(info, count, /*primal=*/true);
  ir_->store_variable(primal_ptr, val);
  if (info.heap_kind != AdStackHeapKind::heap_int) {
    spirv::Value adjoint_ptr = ad_stack_slot_ptr(info, count, /*primal=*/false);
    ir_->store_variable(adjoint_ptr, ir_->get_zero(backing_type));
  }
  ir_->store_variable(info.count_var, ir_->add(count, one));
  // Update the per-task overflow-signal accumulator. `signal = (count >= max) ? stack_id + 1 : 0`; running max
  // across all push sites in this thread. No global memory access.
  spirv::Value any_overflow_var = ensure_any_overflow_signal_var();
  spirv::Value overflow_signal =
      ir_->select(ir_->ge(count, max_val), ir_->uint_immediate_number(ir_->u32_type(), info.stack_id + 1),
                  ir_->uint_immediate_number(ir_->u32_type(), 0));
  spirv::Value prev = ir_->load_variable(any_overflow_var, ir_->u32_type());
  spirv::Value updated = ir_->call_glsl450(ir_->u32_type(), GLSLstd450UMax, prev, overflow_signal);
  ir_->store_variable(any_overflow_var, updated);
}

void TaskCodegen::visit(AdStackPopStmt *stmt) {
  // Intentionally unclamped, unlike the LLVM runtime's stack_pop. The forward AdStackPushStmt now increments
  // count unconditionally (the in-bounds check folded into a clamp + OpSelect there), so push and pop are
  // balanced and count returns to zero at the end of a balanced reverse pass even when intervening pushes
  // overflowed. The LoadTop*/AccAdjoint visitors still clamp idx to max_size-1 so the OpAccessChain stays
  // in-bounds when count is currently above max_size on the overflow path, and the host raises a RuntimeError
  // at the next synchronize() before any garbage adjoint reaches user code.
  auto &info = ad_stacks_.at(stmt->stack);
  spirv::Value count = ir_->load_variable(info.count_var, ir_->u32_type());
  spirv::Value one = ir_->uint_immediate_number(ir_->u32_type(), 1);
  ir_->store_variable(info.count_var, ir_->sub(count, one));
}

// `idx = min(count - 1, max_size - 1)` as a u32 when `clamp_to_max_size` is set. On the overflow path count can
// be above max_size (because the forward push increments count unconditionally and only signals via
// OpAtomicUMax in the bounds-checked build), so the clamp keeps the OpAccessChain in-bounds; without it, hostile
// Vulkan drivers (e.g. Adreno, Mali) TDR on OOB private-memory access before the host-side qd.sync() can raise
// the deferred adstack-overflow exception. The release build (`debug=false`, no overflow signal emitted) trusts
// the published `max_size` and skips both the cap subtract and the UMin call, mirroring LLVM's release-build
// LoadTop emit. `max_size` is a runtime value loaded from AdStackMetadata, so `max_size - 1` becomes an OpISub
// rather than a compile-time immediate when clamping is requested.
static spirv::Value ad_stack_top_index(spirv::IRBuilder *ir,
                                       spirv::Value count,
                                       spirv::Value max_size_val,
                                       bool clamp_to_max_size) {
  spirv::Value one = ir->uint_immediate_number(ir->u32_type(), 1);
  spirv::Value idx = ir->sub(count, one);
  if (!clamp_to_max_size) {
    return idx;
  }
  spirv::Value cap = ir->sub(max_size_val, one);
  return ir->call_glsl450(ir->u32_type(), GLSLstd450UMin, idx, cap);
}

void TaskCodegen::visit(AdStackLoadTopStmt *stmt) {
  // `return_ptr == true` is emitted by ReplaceLocalVarWithStacks::visit(MatrixPtrStmt) when a TensorType
  // loop-carried variable takes a per-element address; the downstream MatrixPtrStmt codegen treats the returned
  // value as a base pointer for OpAccessChain. This path is not implemented here: scalarize with
  // `real_matrix_scalarize=True` (the default) is expected to lower every TensorType adstack to N scalar
  // adstacks + MatrixInit before SPIR-V codegen runs, so no such node reaches this visitor in practice.
  // Raising a targeted error guarantees we surface a clear message if that invariant ever breaks (e.g.
  // `real_matrix_scalarize=False`, or a future scalarize change misses the node type), rather than letting the
  // scalar-load fallthrough register an integer where downstream MatrixPtrStmt expects a pointer.
  QD_ERROR_IF(stmt->return_ptr,
              "SPIR-V codegen does not yet support AdStackLoadTopStmt with return_ptr=true (tensor-typed "
              "loop-carried variable). Ensure scalarize is enabled (real_matrix_scalarize=True) so matrix/vector "
              "adstacks are lowered to scalar ones before codegen.");
  auto &info = ad_stacks_.at(stmt->stack);
  ensure_ad_stack_metadata_loaded(info);
  spirv::Value count = ir_->load_variable(info.count_var, ir_->u32_type());
  spirv::Value idx =
      ad_stack_top_index(ir_.get(), count, info.max_size_val, compile_config_ && (compile_config_->debug));
  spirv::Value ptr = ad_stack_slot_ptr(info, idx, /*primal=*/true);
  spirv::SType backing_type = ad_stack_backing_type(info);
  spirv::Value loaded = ir_->load_variable(ptr, backing_type);
  if (info.elem_type.id != backing_type.id) {
    loaded = ir_->cast(info.elem_type, loaded);  // i32 -> u1 for the heap_int path
  }
  ir_->register_value(stmt->raw_name(), loaded);
}

void TaskCodegen::visit(AdStackLoadTopAdjStmt *stmt) {
  auto &info = ad_stacks_.at(stmt->stack);
  // The auto_diff pass gates AdStackLoadTopAdjStmt/AdStackAccAdjointStmt emission on `is_real` (auto_diff.cpp),
  // so these visitors should never fire on an int-heap-backed stack. If they do, the int heap lacks an adjoint
  // slice and any load would alias the next stack's primal slice - fail loudly instead.
  QD_ASSERT_INFO(info.heap_kind != AdStackHeapKind::heap_int,
                 "AdStackLoadTopAdj on a non-real adstack; autodiff should have suppressed this.");
  ensure_ad_stack_metadata_loaded(info);
  // No elem_type<->backing_type cast here: the primal visitors need it only for the u1 (elem) -> i32 (backing)
  // promotion on the int heap, which this assert excludes. For heap_float the backing type is `info.elem_type`
  // unconditionally, so any cast would be a no-op.
  spirv::Value count = ir_->load_variable(info.count_var, ir_->u32_type());
  spirv::Value idx =
      ad_stack_top_index(ir_.get(), count, info.max_size_val, compile_config_ && (compile_config_->debug));
  spirv::Value ptr = ad_stack_slot_ptr(info, idx, /*primal=*/false);
  spirv::Value loaded = ir_->load_variable(ptr, info.elem_type);
  ir_->register_value(stmt->raw_name(), loaded);
}

void TaskCodegen::visit(AdStackAccAdjointStmt *stmt) {
  auto &info = ad_stacks_.at(stmt->stack);
  QD_ASSERT_INFO(info.heap_kind != AdStackHeapKind::heap_int,
                 "AdStackAccAdjoint on a non-real adstack; autodiff should have suppressed this.");
  ensure_ad_stack_metadata_loaded(info);
  // See the note in `AdStackLoadTopAdjStmt`: no cast is needed because heap_int is excluded by the assert above.
  spirv::Value count = ir_->load_variable(info.count_var, ir_->u32_type());
  spirv::Value idx =
      ad_stack_top_index(ir_.get(), count, info.max_size_val, compile_config_ && (compile_config_->debug));
  spirv::Value ptr = ad_stack_slot_ptr(info, idx, /*primal=*/false);
  spirv::Value old_val = ir_->load_variable(ptr, info.elem_type);
  spirv::Value incr = ir_->query_value(stmt->v->raw_name());
  spirv::Value new_val = ir_->add(old_val, incr);
  ir_->store_variable(ptr, new_val);
}

void TaskCodegen::push_loop_control_labels(spirv::Label continue_label, spirv::Label merge_label) {
  continue_label_stack_.push_back(continue_label);
  merge_label_stack_.push_back(merge_label);
}

void TaskCodegen::pop_loop_control_labels() {
  continue_label_stack_.pop_back();
  merge_label_stack_.pop_back();
}

const spirv::Label TaskCodegen::current_continue_label() const {
  return continue_label_stack_.back();
}

const spirv::Label TaskCodegen::current_merge_label() const {
  return merge_label_stack_.back();
}

const spirv::Label TaskCodegen::return_label() const {
  return continue_label_stack_.front();
}

// Per-thread flag set when SPIRV-Tools reports an ID-space overflow during optimization. The optimizer does not always
// propagate this as `Pass::Status::Failure`, so we capture it here and abort the kernel compilation with a hard error
// (see QD_ERROR_IF in KernelCodegen::run).
static thread_local bool spirv_opt_id_overflow_seen = false;

// Deduplication state for the SPIRV-Tools message consumer. Exposed so that KernelCodegen::run()
// can flush and reset after each optimizer invocation.
static thread_local std::string spirv_msg_last;
static thread_local uint32_t spirv_msg_suppressed = 0;

static void spirv_msg_flush_dedup() {
  if (spirv_msg_suppressed > 0) {
    QD_WARN("(previous SPIRV-Tools message repeated {} more times)", spirv_msg_suppressed);
  }
  spirv_msg_last.clear();
  spirv_msg_suppressed = 0;
}

static void spriv_message_consumer(spv_message_level_t level,
                                   const char *source,
                                   const spv_position_t &position,
                                   const char *message) {
  if (message == nullptr)
    return;
  if (source == nullptr)
    source = "";
  // The raised max_id_bound and intermediate DCE passes are the primary defense. This substring match is fragile
  // (tied to SPIRV-Tools message text). If it stops matching and Run() still returns success with corrupt output
  // (id-0 references), the corrupted SPIR-V will reach the GPU driver. The Run()-failure path is checked
  // independently, but does NOT cover the Run()-succeeds-but-output-is-corrupt case.
  if (std::string_view(message).find("ID overflow") != std::string_view::npos) {
    spirv_opt_id_overflow_seen = true;
  }
  // Deduplicate consecutive identical messages so the log stays readable.
  if (message == spirv_msg_last) {
    ++spirv_msg_suppressed;
    return;
  }
  if (spirv_msg_suppressed > 0) {
    QD_WARN("(previous SPIRV-Tools message repeated {} more times)", spirv_msg_suppressed);
    spirv_msg_suppressed = 0;
  }
  spirv_msg_last = message;
  // TODO: Maybe we can add a macro, e.g. QD_LOG_AT_LEVEL(lv, ...)
  if (level <= SPV_MSG_ERROR) {
    // Log at WARN, not ERROR: QD_ERROR throws, which would propagate through SPIRV-Tools (not
    // exception-safe) and bypass spirv_msg_flush_dedup(). The hard error is raised by QD_ERROR_IF
    // after Run() returns.
    QD_WARN("{}\n[{}:{}:{}] {}", source, position.index, position.line, position.column, message);
  } else if (level <= SPV_MSG_WARNING) {
    QD_WARN("{}\n[{}:{}:{}] {}", source, position.index, position.line, position.column, message);
  } else if (level <= SPV_MSG_INFO) {
    QD_INFO("{}\n[{}:{}:{}] {}", source, position.index, position.line, position.column, message);
  } else if (level <= SPV_MSG_DEBUG) {
    QD_TRACE("{}\n[{}:{}:{}] {}", source, position.index, position.line, position.column, message);
  }
}

KernelCodegen::KernelCodegen(const Params &params) : params_(params), ctx_attribs_(*params.kernel, &params.caps) {
  QD_ASSERT(params.kernel);
  QD_ASSERT(params.ir_root);

  uint32_t spirv_version = params.caps.get(DeviceCapability::spirv_version);

  spv_target_env target_env;
  if (spirv_version >= 0x10600) {
    target_env = SPV_ENV_VULKAN_1_3;
  } else if (spirv_version >= 0x10500) {
    target_env = SPV_ENV_VULKAN_1_2;
  } else if (spirv_version >= 0x10400) {
    target_env = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  } else if (spirv_version >= 0x10300) {
    target_env = SPV_ENV_VULKAN_1_1;
  } else {
    target_env = SPV_ENV_VULKAN_1_0;
  }

  spirv_opt_ = std::make_unique<spvtools::Optimizer>(target_env);
  spirv_opt_->SetMessageConsumer(spriv_message_consumer);
  if (params.enable_spv_opt) {
    // From: SPIRV-Tools/source/opt/optimizer.cpp
    // Intermediate AggressiveDCE passes (matching upstream RegisterPerformancePasses) are critical:
    // without them, dead instructions accumulate between expensive transformations and a single pass
    // (e.g. LocalMultiStoreElim doing SSA construction) can exhaust the SPIR-V ID space.
    spirv_opt_->RegisterPass(spvtools::CreateWrapOpKillPass())
        .RegisterPass(spvtools::CreateDeadBranchElimPass())
        .RegisterPass(spvtools::CreateMergeReturnPass())
        .RegisterPass(spvtools::CreateInlineExhaustivePass())
        .RegisterPass(spvtools::CreateEliminateDeadFunctionsPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreatePrivateToLocalPass())
        .RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(spvtools::CreateLocalSingleStoreElimPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateScalarReplacementPass())
        .RegisterPass(spvtools::CreateLocalAccessChainConvertPass())
        .RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(spvtools::CreateLocalSingleStoreElimPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateLocalMultiStoreElimPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateCCPPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateLoopUnrollPass(true))
        .RegisterPass(spvtools::CreateDeadBranchElimPass())
        .RegisterPass(spvtools::CreateRedundancyEliminationPass())
        .RegisterPass(spvtools::CreateCombineAccessChainsPass())
        .RegisterPass(spvtools::CreateSimplificationPass())
        .RegisterPass(spvtools::CreateSSARewritePass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateVectorDCEPass())
        .RegisterPass(spvtools::CreateDeadInsertElimPass())
        .RegisterPass(spvtools::CreateIfConversionPass())
        .RegisterPass(spvtools::CreateCopyPropagateArraysPass())
        .RegisterPass(spvtools::CreateReduceLoadSizePass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateBlockMergePass())
        .RegisterPass(spvtools::CreateCompactIdsPass());
  }
  spirv_opt_options_.set_run_validator(false);
  // The SPIRV-Tools default ID bound (0x3FFFFF = 4194303) is too low for large autodiff kernels
  // where SSA construction (LocalMultiStoreElim / SSARewrite) creates millions of phi nodes.
  // Raise the optimizer-internal limit; CompactIdsPass at the end of the pipeline renumbers the
  // output back to a dense range (the SPIR-V spec allows IDs up to 2^32-1; 4194303 is only the
  // SPIRV-Tools default, not a spec limit).
  spirv_opt_options_.set_max_id_bound(0x3FFFFFF);

  spirv_tools_ = std::make_unique<spvtools::SpirvTools>(target_env);
}

void KernelCodegen::run(QuadrantsKernelAttributes &kernel_attribs,
                        std::vector<std::vector<uint32_t>> &generated_spirv) {
  auto *root = params_.ir_root->as<Block>();

  const char *dump_ir_env = std::getenv(DUMP_IR_ENV.data());
  bool dump_ir = dump_ir_env != nullptr && std::string(dump_ir_env) == "1";
  std::filesystem::path ir_dump_dir = params_.compile_config->debug_dump_path;
  if (dump_ir) {
    std::filesystem::create_directories(ir_dump_dir);
  }
  if (dump_ir) {
    std::filesystem::path filename = ir_dump_dir / (params_.ti_kernel_name + "_before_final_spirv.ll");
    if (std::ofstream out_file(filename); out_file) {
      std::string outString;
      irpass::print(const_cast<IRNode *>(params_.ir_root), &outString);
      out_file << outString;
    }
  }

  auto &tasks = root->statements;
  for (int i = 0; i < tasks.size(); ++i) {
    TaskCodegen::Params tp;
    tp.task_ir = tasks[i]->as<OffloadedStmt>();
    tp.task_id_in_kernel = i;
    tp.compiled_structs = params_.compiled_structs;
    tp.ctx_attribs = &ctx_attribs_;
    tp.ti_kernel_name = fmt::format("{}_{}", params_.ti_kernel_name, i);
    tp.arch = params_.arch;
    tp.caps = &params_.caps;
    tp.compile_config = params_.compile_config;

    TaskCodegen cgen(tp);
    auto task_res = cgen.run();
    const std::string &spirv_dump_basename = task_res.task_attribs.name;

    std::filesystem::path ir_dump_dir = params_.compile_config->debug_dump_path;
    if (dump_ir) {
      std::string spirv_asm;
      spirv_tools_->Disassemble(task_res.spirv_code, &spirv_asm);
      std::filesystem::path filename = ir_dump_dir / (spirv_dump_basename + "_before_opt.spirv");
      if (std::ofstream out_file(filename); out_file) {
        out_file.write(spirv_asm.c_str(), spirv_asm.size());
      }
    }

    for (auto &[id, access] : task_res.arr_access) {
      for (auto &arr_access_element : ctx_attribs_.arr_access) {
        if (arr_access_element.first == id) {
          arr_access_element.second = arr_access_element.second | access;
        }
      }
    }
    for (auto &[id, access] : task_res.grad_arr_access) {
      for (auto &grad_access_element : ctx_attribs_.grad_arr_access) {
        if (grad_access_element.first == id) {
          grad_access_element.second = grad_access_element.second | access;
        }
      }
    }

    std::vector<uint32_t> optimized_spv(task_res.spirv_code);

    bool success = true;
    {
      spirv_opt_id_overflow_seen = false;
      bool result = false;
      QD_WARN_IF(
          (result = !spirv_opt_->Run(optimized_spv.data(), optimized_spv.size(), &optimized_spv, spirv_opt_options_)),
          "SPIRV optimization failed");
      spirv_msg_flush_dedup();
      if (spirv_opt_id_overflow_seen) {
        QD_WARN("SPIR-V ID overflow detected during optimization of '{}'", tp.ti_kernel_name);
      }
      if (result || spirv_opt_id_overflow_seen) {
        success = false;
      }
    }

    QD_TRACE("SPIRV-Tools-opt: binary size, before={}, after={}", task_res.spirv_code.size(), optimized_spv.size());

    if (dump_ir && success) {
      std::string spirv_asm;
      spirv_tools_->Disassemble(optimized_spv, &spirv_asm);
      std::filesystem::path filename = ir_dump_dir / (spirv_dump_basename + "_after_opt.spirv");
      if (std::ofstream out_file(filename); out_file) {
        out_file.write(spirv_asm.c_str(), spirv_asm.size());
      }
    }

    // Enable to dump SPIR-V assembly of kernels
    if constexpr (false) {
      std::vector<uint32_t> &spirv = success ? optimized_spv : task_res.spirv_code;

      std::string spirv_asm;
      spirv_tools_->Disassemble(spirv, &spirv_asm);
      auto kernel_name = tp.ti_kernel_name;
      QD_WARN("SPIR-V Assembly dump for {} :\n{}\n\n", kernel_name, spirv_asm);

      std::ofstream fout(kernel_name + ".spv", std::ios::binary | std::ios::out);
      fout.write(reinterpret_cast<const char *>(spirv.data()), spirv.size() * sizeof(uint32_t));
      fout.close();
    }

    if (spirv_opt_id_overflow_seen) {
      QD_ERROR_IF(!success,
                  "SPIR-V optimization failed for '{}' due to ID-space overflow. "
                  "The kernel is too large for the SPIRV-Tools optimizer pipeline.",
                  tp.ti_kernel_name);
    } else {
      QD_ERROR_IF(!success, "SPIR-V optimization failed for '{}'.", tp.ti_kernel_name);
    }
    kernel_attribs.tasks_attribs.push_back(std::move(task_res.task_attribs));
    generated_spirv.push_back(std::move(optimized_spv));
  }
  kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
  kernel_attribs.name = params_.ti_kernel_name;
}

}  // namespace spirv
}  // namespace quadrants::lang
