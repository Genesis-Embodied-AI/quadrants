// Note that QD_ASSERT provided by runtime doesn't check fail immediately. As
// a hack, we just check the last error code whenever we call QD_ASSERT.
#define QD_TEST_CHECK(cond, r)                                  \
  do {                                                          \
    QD_ASSERT(cond);                                            \
    if ((r)->error_code) {                                      \
      quadrants_printf((r), "%s", (r)->error_message_template); \
      abort();                                                  \
    }                                                           \
  } while (0)

#define ATOMIC_INSERT(T)                                                     \
  do {                                                                       \
    auto base_ptr = reinterpret_cast<int##T *>(base_ptr_);                   \
    int##T *num_triplets = base_ptr;                                         \
    auto data_base_ptr = base_ptr + 1;                                       \
    auto triplet_id = atomic_add_i##T(num_triplets, 1);                      \
    data_base_ptr[triplet_id * 3] = i;                                       \
    data_base_ptr[triplet_id * 3 + 1] = j;                                   \
    data_base_ptr[triplet_id * 3 + 2] = quadrants_union_cast<int##T>(value); \
  } while (0);

i32 do_nothing(RuntimeContext *context) {
  return 0;
}

i32 refresh_counter(RuntimeContext *context) {
  return 0;
}

i32 insert_triplet_f32(RuntimeContext *context, int64 base_ptr_, int i, int j, float value) {
  ATOMIC_INSERT(32);
  return 0;
}

i32 insert_triplet_f64(RuntimeContext *context, int64 base_ptr_, int i, int j, float64 value) {
  ATOMIC_INSERT(64);
  return 0;
}

i32 test_internal_func_args(RuntimeContext *context, float32 i, float32 j, int32 k) {
  return static_cast<int>((i + j) * k);
}

i32 test_stack(RuntimeContext *context) {
  auto *runtime = context->runtime;
  // Header u64 `n` + max_num_elements * 2 * element_size for primal+adjoint slot pairs. Allocate generously for
  // the guard-case subtests below. Stack-allocated rather than `new u8[...]` to keep the JIT bitcode free of
  // `operator new[]` / `operator delete[]` references that some Linux JIT linker configurations cannot resolve.
  u8 stack_storage[8 + 16 * 2 * 4];
  u8 *stack = stack_storage;
  stack_init(stack);

  // Stash any prior overflow-flag pointer the host has installed and point the runtime at a local slot for
  // the duration of this test. The local slot is host memory in this in-process test fixture, mirroring the
  // pinned-host slot installed by `LlvmRuntimeExecutor::materialize_runtime` in production.
  i64 *prev_flag_dev_ptr = runtime->adstack_overflow_flag_dev_ptr;
  i64 local_flag = 0;
  runtime->adstack_overflow_flag_dev_ptr = &local_flag;

  // Basic push/pop accounting.
  stack_push(runtime, stack, 16, 4, 0);
  stack_push(runtime, stack, 16, 4, 0);
  stack_push(runtime, stack, 16, 4, 0);
  stack_push(runtime, stack, 16, 4, 0);
  QD_TEST_CHECK(*(u64 *)stack == 4, runtime);
  QD_TEST_CHECK(local_flag == 0, runtime);

  // stack_top_primal must point at slot (n - 1) (here: slot 3) when n > 0.
  QD_TEST_CHECK(stack_top_primal(stack, 4) == stack + sizeof(u64) + 3 * 2 * 4, runtime);

  stack_pop(stack);
  stack_pop(stack);
  stack_pop(stack);
  stack_pop(stack);
  QD_TEST_CHECK(*(u64 *)stack == 0, runtime);

  // stack_pop underflow guard: extra pops past n == 0 must not wrap `n` into UINT_MAX. The runtime silently
  // clamps at 0 instead of trapping, so the reverse pass can over-pop without corrupting subsequent kernels.
  stack_pop(stack);
  stack_pop(stack);
  QD_TEST_CHECK(*(u64 *)stack == 0, runtime);

  // stack_top_primal clamping: on an empty stack the top-of-stack pointer must index slot 0 (not `-1`
  // * 2 * element_size, which would point into header territory and crash on read).
  QD_TEST_CHECK(stack_top_primal(stack, 4) == stack + sizeof(u64), runtime);

  // Push past capacity: `n` stops at max_num_elements and the overflow flag flips to 1.
  for (int i = 0; i < 16; i++) {
    stack_push(runtime, stack, 16, 4, 0);
  }
  QD_TEST_CHECK(*(u64 *)stack == 16, runtime);
  QD_TEST_CHECK(local_flag == 0, runtime);
  stack_push(runtime, stack, 16, 4, 0);  // overflow push
  QD_TEST_CHECK(*(u64 *)stack == 16, runtime);
  QD_TEST_CHECK(local_flag == 1, runtime);

  // Restore the prior flag pointer so subsequent tests in the same fixture are not poisoned by our local slot.
  runtime->adstack_overflow_flag_dev_ptr = prev_flag_dev_ptr;
  return 0;
}

i32 test_list_manager(RuntimeContext *context) {
  auto runtime = context->runtime;
  quadrants_printf(runtime, "LLVMRuntime %p\n", runtime);
  auto list = context->runtime->create<ListManager>(runtime, 4, 16);
  for (int i = 0; i < 320; i++) {
    quadrants_printf(runtime, "appending %d\n", i);
    auto j = i + 5;
    list->append(&j);
  }
  for (int i = 0; i < 320; i++) {
    QD_TEST_CHECK(list->get<i32>(i) == i + 5, runtime);
  }
  return 0;
}

i32 test_node_allocator(RuntimeContext *context) {
  auto runtime = context->runtime;
  quadrants_printf(runtime, "LLVMRuntime %p\n", runtime);
  auto nodes = context->runtime->create<NodeManager>(runtime, sizeof(i64), 4);
  Ptr ptrs[24];
  for (int i = 0; i < 19; i++) {
    quadrants_printf(runtime, "allocating %d\n", i);
    ptrs[i] = nodes->allocate();
    quadrants_printf(runtime, "ptr %p\n", ptrs[i]);
  }
  for (int i = 0; i < 5; i++) {
    quadrants_printf(runtime, "deallocating %d\n", i);
    quadrants_printf(runtime, "ptr %p\n", ptrs[i]);
    nodes->recycle(ptrs[i]);
  }
  nodes->gc_serial();
  for (int i = 19; i < 24; i++) {
    quadrants_printf(runtime, "allocating %d\n", i);
    ptrs[i] = nodes->allocate();
  }
  for (int i = 5; i < 19; i++) {
    QD_TEST_CHECK(nodes->locate(ptrs[i]) == i, runtime);
  }

  for (int i = 19; i < 24; i++) {
    auto idx = nodes->locate(ptrs[i]);
    quadrants_printf(runtime, "i %d", i);
    quadrants_printf(runtime, "idx %d", idx);
    QD_TEST_CHECK(idx == i - 19, runtime);
  }
  return 0;
}

i32 test_node_allocator_gc_cpu(RuntimeContext *context) {
  auto runtime = context->runtime;
  quadrants_printf(runtime, "LLVMRuntime %p\n", runtime);
  auto nodes = context->runtime->create<NodeManager>(runtime, sizeof(i64), 4);
  constexpr int kN = 24;
  constexpr int kHalfN = kN / 2;
  Ptr ptrs[kN];
  // Initially |free_list| is empty
  QD_TEST_CHECK(nodes->free_list->size() == 0, runtime);
  for (int i = 0; i < kN; i++) {
    quadrants_printf(runtime, "[1] allocating %d\n", i);
    ptrs[i] = nodes->allocate();
    quadrants_printf(runtime, "[1] ptr %p\n", ptrs[i]);
  }
  for (int i = 0; i < kN; i++) {
    quadrants_printf(runtime, "[1] deallocating %d\n", i);
    quadrants_printf(runtime, "[1] ptr %p\n", ptrs[i]);
    nodes->recycle(ptrs[i]);
  }
  QD_TEST_CHECK(nodes->free_list->size() == 0, runtime);
  nodes->gc_serial();
  // After the first round GC, |free_list| should have |kN| items.
  QD_TEST_CHECK(nodes->free_list->size() == kN, runtime);

  // In the second round, all items should come from |free_list|.
  for (int i = 0; i < kHalfN; i++) {
    quadrants_printf(runtime, "[2] allocating %d\n", i);
    ptrs[i] = nodes->allocate();
    quadrants_printf(runtime, "[2] ptr %p\n", ptrs[i]);
  }
  QD_TEST_CHECK(nodes->free_list_used == kHalfN, runtime);
  for (int i = 0; i < kHalfN; i++) {
    quadrants_printf(runtime, "[2] deallocating %d\n", i);
    quadrants_printf(runtime, "[2] ptr %p\n", ptrs[i]);
    nodes->recycle(ptrs[i]);
  }
  nodes->gc_serial();
  // After GC, all items should be returned to |free_list|.
  quadrants_printf(runtime, "free_list_size=%d\n", nodes->free_list->size());
  QD_TEST_CHECK(nodes->free_list->size() == kN, runtime);

  return 0;
}

i32 test_active_mask(RuntimeContext *context) {
  auto rt = context->runtime;
  quadrants_printf(rt, "%d activemask %x\n", thread_idx(), cuda_active_mask());

  auto active_mask = cuda_active_mask();
  auto remaining = active_mask;
  while (remaining) {
    auto leader = cttz_i32(remaining);
    quadrants_printf(rt, "current leader %d bid %d tid %d\n", leader, block_idx(), thread_idx());
    warp_barrier(active_mask);
    remaining &= ~(1u << leader);
  }

  return 0;
}

i32 test_shfl(RuntimeContext *context) {
  auto rt = context->runtime;
  auto s = cuda_shfl_down_sync_i32(cuda_active_mask(), warp_idx() + 1000, 2, 31);
  quadrants_printf(rt, "tid %d tid_shfl %d\n", thread_idx(), s);

  return 0;
}

#undef QD_TEST_CHECK
