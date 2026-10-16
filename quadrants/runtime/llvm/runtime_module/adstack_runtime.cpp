// Included by runtime.cpp after LLVMRuntime and RuntimeContext definitions.
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer);

extern "C" i32 adstack_measure_peak(RuntimeContext *ctx, i64 depth) {
  auto *slot = ctx->adstack_measurement;
  i64 previous = __atomic_load_n(slot, __ATOMIC_RELAXED);
  while (previous < depth &&
         !__atomic_compare_exchange_n(slot, &previous, depth, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
  return 0;
}

extern "C" u64 adstack_checked_count(RuntimeContext *ctx, u64 required) {
  auto *flag = ctx->adstack_measurement + 1;
  if (required > ctx->adstack_capacity) {
    __atomic_store_n(flag, i64(1), __ATOMIC_RELAXED);
    return ctx->adstack_capacity;
  }
  // No thread exits early: reductions and barriers must still execute after a failure. The host discards
  // the entire result. Avoid a global atomic load on every successful push.
  return required;
}

extern "C" u64 adstack_capacity(RuntimeContext *ctx) {
  return ctx->adstack_capacity;
}

extern "C" void runtime_get_adstack_heap_field_ptr(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_heap_buffer);
}

extern "C" void adstack_runtime_zero_init(LLVMRuntime *runtime) {
  runtime->adstack_heap_buffer = nullptr;
}
