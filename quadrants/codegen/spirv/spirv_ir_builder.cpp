#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "fp16.h"
#include <cstdint>
#include <iterator>

#include "spirv-tools/linker.hpp"
#include "workgroup_spv.h"

namespace quadrants::lang {

namespace spirv {

// Link the kernel with the supplied compiled GLSL helper libraries, resolving imported functions to their
// implementations. Return the combined SPIR-V module, or report an error if linking fails.
std::vector<uint32_t> IRBuilder::link_shader_helpers(const std::vector<uint32_t> &kernel,
                                                     std::vector<std::vector<uint32_t>> libraries) {
  // These helper libraries use Input and Function pointers. Match their addressing model to the kernel.
  // Any required physical-storage capability and extension must already be declared by the kernel.
  uint32_t addressing_model = get_module_addressing_model(kernel);
  for (auto &library : libraries) {
    set_module_addressing_model(library, addressing_model);
  }
  // Accept inputs through SPIR-V 1.6; SetUseHighestVersion below keeps the output at the highest input version,
  // so a 1.5 kernel linked with our 1.0 helper stays at 1.5 and does not require a device that supports 1.6.
  spvtools::Context context(SPV_ENV_UNIVERSAL_1_6);
  std::string error;
  context.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
    error += message;
    error += '\n';
  });
  spvtools::LinkerOptions options;
  options.SetUseHighestVersion(true);
  std::vector<uint32_t> linked;
  libraries.insert(libraries.begin(), kernel);
  auto result = spvtools::Link(context, libraries, &linked, options);
  QD_ERROR_IF(result != SPV_SUCCESS, "Failed to link GLSL shader helpers: {}", error);
  return linked;
}

// Return the module's addressing model, defaulting to Logical if OpMemoryModel is absent.
uint32_t IRBuilder::get_module_addressing_model(const std::vector<uint32_t> &spirv_module) {
  // OpMemoryModel has two operands: 0 is the addressing model (how pointers are represented), and 1 is the memory
  // model (rules for memory operations). For example, OpMemoryModel Logical GLSL450 uses Logical addressing and
  // GLSL450 memory rules. Read and update operand 0 to match the libraries' addressing model to the kernel's.
  return get_instruction_operand(spirv_module, spv::OpMemoryModel, /* operand_index= */ 0,
                                 /* default_value= */ spv::AddressingModelLogical);
}

// Update the module's addressing model if OpMemoryModel is present.
void IRBuilder::set_module_addressing_model(std::vector<uint32_t> &spirv_module, uint32_t addressing_model) {
  set_instruction_operand(spirv_module, spv::OpMemoryModel, /* operand_index= */ 0, addressing_model);
}

// Read a zero-based operand of the first matching instruction, or return default_value if the instruction is absent.
uint32_t IRBuilder::get_instruction_operand(const std::vector<uint32_t> &spirv_module,
                                            spv::Op opcode,
                                            size_t operand_index,
                                            uint32_t default_value) {
  size_t instruction_index = find_instruction(spirv_module, opcode);
  if (instruction_index == spirv_module.size()) {
    return default_value;
  }
  // Add one because the instruction's word count includes its first word, which holds the opcode and word count.
  QD_ASSERT(operand_index + 1 < (spirv_module[instruction_index] >> 16));
  return spirv_module.at(instruction_index + 1 + operand_index);
}

// Update a zero-based operand of the first matching instruction; leave the module unchanged if it is absent.
void IRBuilder::set_instruction_operand(std::vector<uint32_t> &spirv_module,
                                        spv::Op opcode,
                                        size_t operand_index,
                                        uint32_t value) {
  size_t instruction_index = find_instruction(spirv_module, opcode);
  if (instruction_index == spirv_module.size()) {
    return;
  }
  // Add one because the instruction's word count includes its first word, which holds the opcode and word count.
  QD_ASSERT(operand_index + 1 < (spirv_module[instruction_index] >> 16));
  spirv_module.at(instruction_index + 1 + operand_index) = value;
}

// Return the word index of the first matching instruction, or spirv_module.size() if it is absent.
size_t IRBuilder::find_instruction(const std::vector<uint32_t> &spirv_module, spv::Op opcode) {
  // Skip the five-word module header. Each instruction encodes its word count above its 16-bit opcode.
  for (size_t i = 5; i < spirv_module.size(); i += spirv_module[i] >> 16) {
    if ((spirv_module[i] & 0xffff) == opcode) {
      return i;
    }
  }
  return spirv_module.size();
}

using cap = DeviceCapability;

void IRBuilder::init_header() {
  QD_ASSERT(header_.size() == 0U);
  header_.push_back(spv::MagicNumber);

  header_.push_back(caps_->get(cap::spirv_version));

  QD_TRACE("SPIR-V Version {}", caps_->get(cap::spirv_version));

  // generator: set to 0, unknown
  header_.push_back(0U);
  // Bound: set during Finalize
  header_.push_back(0U);
  // Schema: reserved
  header_.push_back(0U);

  // capability
  ib_.begin(spv::OpCapability).add(spv::CapabilityShader).commit(&capabilities_extensions_imports_);

  if (caps_->get(cap::spirv_has_atomic_float64_add)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityAtomicFloat64AddEXT).commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_atomic_float_add)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityAtomicFloat32AddEXT).commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_atomic_float_minmax)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityAtomicFloat32MinMaxEXT).commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_variable_ptr)) {
    /*
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityVariablePointers)
        .commit(&capabilities_extensions_imports_);
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityVariablePointersStorageBuffer)
        .commit(&capabilities_extensions_imports_);
        */
  }

  if (caps_->get(cap::spirv_has_int8)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt8).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_int16)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt16).commit(&capabilities_extensions_imports_);
  }
  // `CapabilityStorageBuffer{8,16}BitAccess` gate narrow-typed loads / stores through a
  // descriptor-bound `StorageBuffer` pointer (e.g. `OpLoad %_ptr_StorageBuffer_ushort`). The
  // existing codegen has been emitting these narrow-typed accesses via the uint-punning path in
  // `load_buffer` / `store_buffer` for a while (`get_quadrants_uint_type(i16) = u16`,
  // `get_quadrants_uint_type(i8) = u8`), which strict Vulkan validation requires these capabilities
  // for -- so we emit them unconditionally whenever the queried feature is set, independent of
  // whether the current kernel actually uses a narrow type. The cost is a single extra
  // `OpCapability` word in the shader header; the upside is that every 16-bit / 8-bit field or
  // ndarray access is spec-compliant on drivers that enforce the letter of
  // `SPV_KHR_{8,16}bit_storage`.
  if (caps_->get(cap::spirv_has_storage_buffer_8bit_access)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityStorageBuffer8BitAccess).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_storage_buffer_16bit_access)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityStorageBuffer16BitAccess).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_int64)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt64).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_atomic_int64)) {
    // Required for OpAtomicLoad/OpAtomicCompareExchange on u64, used by
    // the CAS-based f64 shared float atomic emulation path.
    ib_.begin(spv::OpCapability).add(spv::CapabilityInt64Atomics).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_float16)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityFloat16).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_float64)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityFloat64).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_physical_storage_buffer)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityPhysicalStorageBufferAddresses)
        .commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_shader_clock)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityShaderClockKHR).commit(&capabilities_extensions_imports_);
  }

  // Subgroup / GroupNonUniform capabilities. Required by every SPIR-V module that lowers any
  // `qd.simt.subgroup.*` or `qd.simt.block.*` op (the new QIPC subgroup/block work in #676/#684 has
  // significantly broadened how often these are emitted). Strict Vulkan validation rejects
  // `OpGroupNonUniform*` and the `SubgroupLocalInvocationId` BuiltIn unless these caps are declared
  // — and drivers that tolerated their absence at i32 width still silently produce wrong results
  // for i64 ops without them. Emit unconditionally whenever the device exposes the corresponding
  // subgroup feature; the underlying caps are gated by Vulkan's
  // `VkPhysicalDeviceSubgroupProperties::supportedOperations` query in `vulkan_device_creator.cpp`.
  if (caps_->get(cap::spirv_has_subgroup_basic)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityGroupNonUniform).commit(&capabilities_extensions_imports_);
    // `Broadcast` / `Shuffle` and the relative variants used by `_exclusive_scan_tiled` / shuffle
    // intrinsics. The two are separate SPIR-V caps but every desktop/mobile Vulkan implementation
    // that advertises basic GroupNonUniform also advertises both shuffle variants in practice
    // (and the SPIR-V spec marks both as required for any `OpGroupNonUniformShuffle{,Up,Down}` /
    // `OpGroupNonUniformBroadcast` emission), so we tie them to `spirv_has_subgroup_basic` rather
    // than introducing a separate device cap.
    //
    // FIXME: Vulkan's `VkPhysicalDeviceSubgroupProperties::supportedOperations` exposes
    // `VK_SUBGROUP_FEATURE_SHUFFLE_BIT` / `VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT` as bits separate
    // from `VK_SUBGROUP_FEATURE_BASIC_BIT`, and the spec permits a conformant device to advertise
    // BASIC without SHUFFLE. A strict validator on such a device would reject every Quadrants
    // SPIR-V module here — even kernels that do not actually use shuffle — because the declared
    // capability is unsupported. No such device has been observed in the wild (the codepath this
    // change broadens was already exposed pre-PR for any kernel emitting
    // `OpGroupNonUniformShuffle*`), so this PR does not make it worse, but it does extend the
    // surface. Fix: introduce `spirv_has_subgroup_shuffle{,_relative}` device caps in
    // `rhi/rhi_constants.inc.h`, populate them from the two Vulkan bits in
    // `vulkan_device_creator.cpp::populate_subgroup_caps`, and gate the two
    // `CapabilityGroupNonUniformShuffle{,Relative}` emissions on those caps individually.
    ib_.begin(spv::OpCapability).add(spv::CapabilityGroupNonUniformShuffle).commit(&capabilities_extensions_imports_);
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityGroupNonUniformShuffleRelative)
        .commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_subgroup_vote)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityGroupNonUniformVote).commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_subgroup_arithmetic)) {
    ib_.begin(spv::OpCapability)
        .add(spv::CapabilityGroupNonUniformArithmetic)
        .commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_subgroup_ballot)) {
    ib_.begin(spv::OpCapability).add(spv::CapabilityGroupNonUniformBallot).commit(&capabilities_extensions_imports_);
  }

  ib_.begin(spv::OpExtension).add("SPV_KHR_storage_buffer_storage_class").commit(&capabilities_extensions_imports_);

  // `SPV_KHR_{8,16}bit_storage` is paired with `CapabilityStorageBuffer{8,16}BitAccess` above.
  // Both the capability and the extension are needed for narrow-typed `StorageBuffer` loads /
  // stores to validate on Vulkan; declaring only the capability without the extension is
  // ill-formed SPIR-V.
  if (caps_->get(cap::spirv_has_storage_buffer_8bit_access)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_8bit_storage").commit(&capabilities_extensions_imports_);
  }
  if (caps_->get(cap::spirv_has_storage_buffer_16bit_access)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_16bit_storage").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_no_integer_wrap_decoration)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_no_integer_wrap_decoration").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_non_semantic_info)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_non_semantic_info").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_variable_ptr)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_variable_pointers").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_atomic_float_add)) {
    ib_.begin(spv::OpExtension).add("SPV_EXT_shader_atomic_float_add").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_atomic_float_minmax)) {
    ib_.begin(spv::OpExtension).add("SPV_EXT_shader_atomic_float_min_max").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_shader_clock)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_shader_clock").commit(&capabilities_extensions_imports_);
  }

  if (caps_->get(cap::spirv_has_physical_storage_buffer)) {
    ib_.begin(spv::OpExtension).add("SPV_KHR_physical_storage_buffer").commit(&capabilities_extensions_imports_);

    // memory model
    ib_.begin(spv::OpMemoryModel)
        .add_seq(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450)
        .commit(&entry_);
  } else {
    ib_.begin(spv::OpMemoryModel).add_seq(spv::AddressingModelLogical, spv::MemoryModelGLSL450).commit(&entry_);
  }

  this->init_pre_defs();
}

std::vector<uint32_t> IRBuilder::finalize() {
  // SPIR-V module layout, in order (each element is a 32-bit word):
  // 1. Five-word header: magic number, version, generator ID, ID bound, reserved word.
  // 2. Required capabilities, extensions, and extended-instruction-set imports.
  // 3. Memory model, entry points, and execution modes.
  // 4. Debug information and annotations.
  // 5. Types, constants, and global variables.
  // 6. Function declarations without bodies, then function definitions with bodies.
  std::vector<uint32_t> spirv_module;

  // 1. Five-word header: magic number, version, generator ID, ID bound, reserved word.
  const int bound_loc = 3;
  header_[bound_loc] = id_counter_;
  spirv_module.insert(spirv_module.end(), header_.begin(), header_.end());

  // 2. Required capabilities, extensions, and extended-instruction-set imports.
  if (!imported_glsl_function_declarations_.empty()) {
    spirv_module.insert(spirv_module.end(), {(2u << 16) | spv::OpCapability, spv::CapabilityLinkage});
  }
  spirv_module.insert(spirv_module.end(), capabilities_extensions_imports_.begin(),
                      capabilities_extensions_imports_.end());

  // 3. Memory model, entry points, and execution modes.
  spirv_module.insert(spirv_module.end(), entry_.begin(), entry_.end());
  spirv_module.insert(spirv_module.end(), exec_mode_.begin(), exec_mode_.end());

  // 4. Debug information and annotations.
  spirv_module.insert(spirv_module.end(), strings_.begin(), strings_.end());
  spirv_module.insert(spirv_module.end(), names_.begin(), names_.end());
  spirv_module.insert(spirv_module.end(), decorate_.begin(), decorate_.end());

  // 5. Types, constants, and global variables.
  spirv_module.insert(spirv_module.end(), global_.begin(), global_.end());

  // 6. Function declarations without bodies, then function definitions with bodies.
  spirv_module.insert(spirv_module.end(), imported_glsl_function_declarations_.begin(),
                      imported_glsl_function_declarations_.end());
  spirv_module.insert(spirv_module.end(), func_header_.begin(), func_header_.end());
  spirv_module.insert(spirv_module.end(), function_.begin(), function_.end());

  // Link the completed module.
  if (!imported_glsl_function_declarations_.empty()) {
    std::vector<uint32_t> workgroup_library(std::begin(workgroup_helper_spv), std::end(workgroup_helper_spv));
    return link_shader_helpers(spirv_module, {std::move(workgroup_library)});
  }
  return spirv_module;
}

void IRBuilder::init_pre_defs() {
  ext_glsl450_ = ext_inst_import("GLSL.std.450");
  t_bool_ = declare_primitive_type(get_data_type<bool>());
  if (caps_->get(cap::spirv_has_int8)) {
    t_int8_ = declare_primitive_type(get_data_type<int8>());
    t_uint8_ = declare_primitive_type(get_data_type<uint8>());
  }
  if (caps_->get(cap::spirv_has_int16)) {
    t_int16_ = declare_primitive_type(get_data_type<int16>());
    t_uint16_ = declare_primitive_type(get_data_type<uint16>());
  }
  t_int32_ = declare_primitive_type(get_data_type<int32>());
  t_uint32_ = declare_primitive_type(get_data_type<uint32>());
  if (caps_->get(cap::spirv_has_int64)) {
    t_int64_ = declare_primitive_type(get_data_type<int64>());
    t_uint64_ = declare_primitive_type(get_data_type<uint64>());
  }
  t_fp32_ = declare_primitive_type(get_data_type<float32>());
  if (caps_->get(cap::spirv_has_float16)) {
    t_fp16_ = declare_primitive_type(PrimitiveType::f16);
  }
  if (caps_->get(cap::spirv_has_float64)) {
    t_fp64_ = declare_primitive_type(get_data_type<float64>());
  }
  // declare void, and void functions
  t_void_.id = id_counter_++;
  ib_.begin(spv::OpTypeVoid).add(t_void_).commit(&global_);
  t_void_func_.id = id_counter_++;
  ib_.begin(spv::OpTypeFunction).add_seq(t_void_func_, t_void_).commit(&global_);

  // compute shader related types
  t_v2_int_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v2_int_).add_seq(t_int32_, 2).commit(&global_);

  t_v3_int_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v3_int_).add_seq(t_int32_, 3).commit(&global_);

  t_v3_uint_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v3_uint_).add_seq(t_uint32_, 3).commit(&global_);

  t_v4_uint_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v4_uint_).add_seq(t_uint32_, 4).commit(&global_);

  t_v4_fp32_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v4_fp32_).add_seq(t_fp32_, 4).commit(&global_);

  t_v2_fp32_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v2_fp32_).add_seq(t_fp32_, 2).commit(&global_);

  t_v3_fp32_.id = id_counter_++;
  ib_.begin(spv::OpTypeVector).add(t_v3_fp32_).add_seq(t_fp32_, 3).commit(&global_);

  // pre-defined constants
  const_i32_zero_ = int_immediate_number(t_int32_, 0);
  const_i32_one_ = int_immediate_number(t_int32_, 1);
}

Value IRBuilder::debug_string(std::string s) {
  Value val = new_value(SType(), ValueKind::kNormal);
  ib_.begin(spv::OpString).add_seq(val, s).commit(&strings_);
  return val;
}

PhiValue IRBuilder::make_phi(const SType &out_type, uint32_t num_incoming) {
  Value val = new_value(out_type, ValueKind::kNormal);
  ib_.begin(spv::OpPhi).add_seq(out_type, val);
  for (uint32_t i = 0; i < 2 * num_incoming; ++i) {
    ib_.add(0);
  }

  PhiValue phi;
  phi.id = val.id;
  phi.stype = out_type;
  phi.flag = ValueKind::kNormal;
  phi.instr = ib_.commit(&function_);
  return phi;
}

Value IRBuilder::int_immediate_number(const SType &dtype, int64_t value, bool cache) {
  QD_ASSERT(is_integral(dtype.dt));
  return get_const(dtype, reinterpret_cast<uint64_t *>(&value), cache);
}

Value IRBuilder::uint_immediate_number(const SType &dtype, uint64_t value, bool cache) {
  QD_ASSERT(is_integral(dtype.dt));
  return get_const(dtype, &value, cache);
}

Value IRBuilder::float_immediate_number(const SType &dtype, double value, bool cache) {
  QD_ASSERT(is_real(dtype.dt));
  if (data_type_bits(dtype.dt) == 64) {
    return get_const(dtype, reinterpret_cast<uint64_t *>(&value), cache);
  } else if (data_type_bits(dtype.dt) == 32) {
    float fvalue = static_cast<float>(value);
    uint32_t *ptr = reinterpret_cast<uint32_t *>(&fvalue);
    uint64_t data = ptr[0];
    return get_const(dtype, &data, cache);
  } else if (data_type_bits(dtype.dt) == 16) {
    float fvalue = static_cast<float>(value);
    uint64_t data = fp16_ieee_from_fp32_value(fvalue);
    return get_const(dtype, &data, cache);
  } else {
    QD_ERROR("Type {} not supported.", dtype.dt->to_string());
  }
}

SType IRBuilder::get_null_type() {
  SType res;
  res.id = id_counter_++;
  return res;
}

SType IRBuilder::get_primitive_type(const DataType &dt) const {
  if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return t_bool_;
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    if (!caps_->get(cap::spirv_has_float16))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_fp16_;
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return t_fp32_;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    if (!caps_->get(cap::spirv_has_float64))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_fp64_;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    if (!caps_->get(cap::spirv_has_int8))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_int8_;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    if (!caps_->get(cap::spirv_has_int16))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_int16_;
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return t_int32_;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    if (!caps_->get(cap::spirv_has_int64))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_int64_;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    if (!caps_->get(cap::spirv_has_int8))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_uint8_;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    if (!caps_->get(cap::spirv_has_int16))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_uint16_;
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return t_uint32_;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    if (!caps_->get(cap::spirv_has_int64))
      QD_ERROR("Type {} not supported.", dt->to_string());
    return t_uint64_;
  } else {
    QD_ERROR("Type {} not supported.", dt->to_string());
  }
}

SType IRBuilder::from_quadrants_type(const DataType &dt, bool has_buffer_ptr) {
  if (dt->is<PrimitiveType>()) {
    return get_primitive_type(dt);
  } else if (dt->is<PointerType>()) {
    if (has_buffer_ptr) {
      return t_uint64_;
    } else {
      return t_uint32_;
    }
  } else if (auto struct_type = dt->cast<lang::StructType>()) {
    std::vector<std::tuple<SType, std::string, size_t>> components;
    for (const auto &[type, name, offset] : struct_type->elements()) {
      components.push_back(std::make_tuple(from_quadrants_type(type, has_buffer_ptr), name, offset));
    }
    return create_struct_type(components);
  } else {
    QD_ERROR("Type {} not supported.", dt->to_string());
  }
}

size_t IRBuilder::get_primitive_type_size(const DataType &dt) const {
  if (!dt->is<PrimitiveType>()) {
    QD_ERROR("Type {} not supported.", dt->to_string());
  }
  if (dt == PrimitiveType::i64 || dt == PrimitiveType::u64 || dt == PrimitiveType::f64) {
    return 8;
  } else if (dt == PrimitiveType::i32 || dt == PrimitiveType::u32 || dt == PrimitiveType::f32) {
    return 4;
  } else if (dt == PrimitiveType::i16 || dt == PrimitiveType::u16 || dt == PrimitiveType::f16) {
    return 2;
  } else {
    return 1;
  }
}

SType IRBuilder::get_primitive_uint_type(const DataType &dt) const {
  if (dt == PrimitiveType::i64 || dt == PrimitiveType::u64 || dt == PrimitiveType::f64) {
    return t_uint64_;
  } else if (dt == PrimitiveType::i32 || dt == PrimitiveType::u32 || dt == PrimitiveType::f32) {
    return t_uint32_;
  } else if (dt == PrimitiveType::i16 || dt == PrimitiveType::u16 || dt == PrimitiveType::f16) {
    return t_uint16_;
  } else if (dt == PrimitiveType::u1) {
    return t_bool_;
  } else {
    return t_uint8_;
  }
}

DataType IRBuilder::get_quadrants_uint_type(const DataType &dt) const {
  if (dt == PrimitiveType::i64 || dt == PrimitiveType::u64 || dt == PrimitiveType::f64) {
    return PrimitiveType::u64;
  } else if (dt == PrimitiveType::i32 || dt == PrimitiveType::u32 || dt == PrimitiveType::f32) {
    return PrimitiveType::u32;
  } else if (dt == PrimitiveType::i16 || dt == PrimitiveType::u16 || dt == PrimitiveType::f16) {
    return PrimitiveType::u16;
  } else if (dt == PrimitiveType::u1) {
    return PrimitiveType::u1;
  } else {
    return PrimitiveType::u8;
  }
}

SType IRBuilder::get_pointer_type(const SType &value_type, spv::StorageClass storage_class) {
  auto key = std::make_pair(value_type.id, storage_class);
  auto it = pointer_type_tbl_.find(key);
  if (it != pointer_type_tbl_.end()) {
    return it->second;
  }
  SType t;
  t.id = id_counter_++;
  t.flag = TypeKind::kPtr;
  t.element_type_id = value_type.id;
  t.storage_class = storage_class;
  ib_.begin(spv::OpTypePointer).add_seq(t, storage_class, value_type).commit(&global_);
  // An `OpTypePointer` in the `PhysicalStorageBuffer` storage class that points to a scalar or vector
  // needs an explicit `ArrayStride` decoration for `OpPtrAccessChain`'s `Element` offset to be scaled
  // correctly on Vulkan. Without the decoration, drivers that strictly follow SPV_KHR_physical_storage_buffer
  // treat the stride as undefined and collapse every element index to the base address, which manifests as
  // `arr[i]` reads returning `arr[0]` for all `i` across the whole kernel (and any indexed ndarray write
  // landing on slot 0). Sized-pointees (structs, arrays) already carry explicit layout decorations so they
  // don't need this; the fix is limited to scalars/vectors, where the natural stride is just the pointee
  // byte size. Uniform / StorageBuffer / Input / Output / Workgroup pointers don't use PSB arithmetic, so
  // the decoration is a no-op for them and is skipped.
  if (storage_class == spv::StorageClassPhysicalStorageBuffer && value_type.flag == TypeKind::kPrimitive) {
    size_t stride = get_primitive_type_size(value_type.dt);
    if (stride > 0) {
      this->decorate(spv::OpDecorate, t, spv::DecorationArrayStride, uint32_t(stride));
    }
  }
  pointer_type_tbl_[key] = t;
  return t;
}

SType IRBuilder::get_storage_pointer_type(const SType &value_type) {
  spv::StorageClass storage_class;
  if (caps_->get(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  return get_pointer_type(value_type, storage_class);
}

SType IRBuilder::get_function_array_type(const SType &_value_type, uint32_t num_elems) {
  auto value_type = _value_type;
  if (value_type.dt->is_primitive(PrimitiveTypeID::u1)) {
    value_type = i32_type();
  }
  SType arr_type;
  arr_type.id = id_counter_++;
  arr_type.flag = TypeKind::kPtr;
  arr_type.element_type_id = value_type.id;

  if (num_elems != 0) {
    Value length = uint_immediate_number(t_uint32_, num_elems);
    ib_.begin(spv::OpTypeArray).add_seq(arr_type, value_type, length).commit(&global_);
  } else {
    ib_.begin(spv::OpTypeRuntimeArray).add_seq(arr_type, value_type).commit(&global_);
  }

  return arr_type;
}

SType IRBuilder::get_array_type(const SType &_value_type, uint32_t num_elems) {
  // Identical bookkeeping to `get_function_array_type` plus the `ArrayStride` decoration the storage-buffer
  // / PSB / Uniform interface requires. Delegate the `OpTypeArray` emission to keep the two in sync, then
  // add the decoration on top.
  SType arr_type = get_function_array_type(_value_type, num_elems);

  // Mirror `get_function_array_type`'s `u1 -> i32` rewrite so the stride below matches the `OpTypeArray`
  // element type (`bool` is 1-byte on every host but the array is emitted with `i32` elements; without this
  // rewrite the stride would land on `1` and `spirv-val` rejects `ArrayStride < element_size`).
  auto value_type = _value_type;
  if (value_type.dt->is_primitive(PrimitiveTypeID::u1)) {
    value_type = i32_type();
  }

  uint32_t nbytes;
  if (value_type.flag == TypeKind::kPrimitive) {
    const auto nbits = data_type_bits(value_type.dt);
    nbytes = static_cast<uint32_t>(nbits) / 8;
  } else if (value_type.flag == TypeKind::kSNodeStruct) {
    nbytes = value_type.snode_desc.container_stride;
  } else {
    QD_ERROR("buffer type must be primitive or snode struct");
  }

  if (nbytes == 0) {
    if (value_type.flag == TypeKind::kPrimitive) {
      QD_WARN("Invalid primitive bit size");
    } else {
      QD_WARN("Invalid container stride");
    }
  }

  // decorate the array type
  this->decorate(spv::OpDecorate, arr_type, spv::DecorationArrayStride, nbytes);

  return arr_type;
}

SType IRBuilder::get_struct_array_type(const SType &value_type, uint32_t num_elems) {
  SType arr_type = get_array_type(value_type, num_elems);

  // declare struct of array
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.flag = TypeKind::kStruct;
  struct_type.element_type_id = value_type.id;
  ib_.begin(spv::OpTypeStruct).add_seq(struct_type, arr_type).commit(&global_);
  // decorate the array type.
  ib_.begin(spv::OpMemberDecorate).add_seq(struct_type, 0, spv::DecorationOffset, 0).commit(&decorate_);

  if (caps_->get(cap::spirv_version) < 0x10300) {
    // NOTE: BufferBlock was deprecated in SPIRV 1.3
    // use StorageClassStorageBuffer instead.
    // runtime array are always decorated as BufferBlock(shader storage buffer)
    if (num_elems == 0) {
      this->decorate(spv::OpDecorate, struct_type, spv::DecorationBufferBlock);
    }
  } else {
    this->decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);
  }

  return struct_type;
}

SType IRBuilder::create_struct_type(std::vector<std::tuple<SType, std::string, size_t>> &components) {
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.flag = TypeKind::kStruct;

  auto &builder = ib_.begin(spv::OpTypeStruct).add_seq(struct_type);

  for (auto &[type, name, offset] : components) {
    builder.add_seq(type);
  }

  builder.commit(&global_);

  int i = 0;
  for (auto &[type, name, offset] : components) {
    this->decorate(spv::OpMemberDecorate, struct_type, i, spv::DecorationOffset, offset);
    this->debug_name(spv::OpMemberName, struct_type, i, name);
    i++;
  }

  return struct_type;
}

Value IRBuilder::buffer_struct_argument(const SType &struct_type,
                                        uint32_t descriptor_set,
                                        uint32_t binding,
                                        const std::string &name) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class;
  if (caps_->get(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  this->debug_name(spv::OpName, struct_type, name + "_t");

  if (caps_->get(cap::spirv_version) < 0x10300) {
    // NOTE: BufferBlock was deprecated in SPIRV 1.3
    // use StorageClassStorageBuffer instead.
    // runtime array are always decorated as BufferBlock(shader storage buffer)
    this->decorate(spv::OpDecorate, struct_type, spv::DecorationBufferBlock);
  } else {
    this->decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);
  }

  SType ptr_type = get_pointer_type(struct_type, storage_class);

  this->debug_name(spv::OpName, ptr_type, name + "_ptr");

  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable).add_seq(ptr_type, val, storage_class).commit(&global_);

  this->debug_name(spv::OpName, val, name);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet, descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::uniform_struct_argument(const SType &struct_type,
                                         uint32_t descriptor_set,
                                         uint32_t binding,
                                         const std::string &name) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class = spv::StorageClassUniform;

  this->debug_name(spv::OpName, struct_type, name + "_t");

  this->decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);

  SType ptr_type = get_pointer_type(struct_type, storage_class);

  this->debug_name(spv::OpName, ptr_type, name + "_ptr");

  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable).add_seq(ptr_type, val, storage_class).commit(&global_);

  this->debug_name(spv::OpName, val, name);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet, descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::buffer_argument(const SType &value_type,
                                 uint32_t descriptor_set,
                                 uint32_t binding,
                                 const std::string &name) {
  // NOTE: BufferBlock was deprecated in SPIRV 1.3
  // use StorageClassStorageBuffer instead.
  spv::StorageClass storage_class;
  if (caps_->get(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  SType sarr_type = get_struct_array_type(value_type, 0);

  auto typed_name = name + "_" + value_type.dt.to_string();

  this->debug_name(spv::OpName, sarr_type, typed_name + "_struct_array");

  SType ptr_type = get_pointer_type(sarr_type, storage_class);

  this->debug_name(spv::OpName, sarr_type, typed_name + "_ptr");

  Value val = new_value(ptr_type, ValueKind::kStructArrayPtr);
  ib_.begin(spv::OpVariable).add_seq(ptr_type, val, storage_class).commit(&global_);

  this->debug_name(spv::OpName, val, typed_name);

  this->decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet, descriptor_set);
  this->decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::struct_array_access(const SType &res_type, Value buffer, Value index) {
  QD_ASSERT(buffer.flag == ValueKind::kStructArrayPtr);
  QD_ASSERT(res_type.flag == TypeKind::kPrimitive);

  spv::StorageClass storage_class;
  if (caps_->get(cap::spirv_version) < 0x10300) {
    storage_class = spv::StorageClassUniform;
  } else {
    storage_class = spv::StorageClassStorageBuffer;
  }

  SType ptr_type = this->get_pointer_type(res_type, storage_class);
  Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpAccessChain).add_seq(ptr_type, ret, buffer, const_i32_zero_, index).commit(&function_);

  return ret;
}

// Emit a call to a named GLSL function taking and returning a uint32, and return the value representing its result.
//
// glslang represents this GLSL value parameter as a pointer in SPIR-V. SPIR-V also supports value parameters; the
// pointer representation is glslang's choice. The caller must match the compiled helper. Equivalent C++-style
// pseudocode:
//
// GLSL source, with an illustrative caller:
//   uint get_work_group_id(uint dim_index) {
//     return gl_WorkGroupID[dim_index];
//   }
//
//   void caller() {
//     uint result = get_work_group_id(0);
//   }
//
// Equivalent pointer passing, not literal generated source:
//   uint get_work_group_id(uint* dim_index_ptr) {
//     return gl_WorkGroupID[*dim_index_ptr];
//   }
//
//   void caller() {
//     uint argument = 0;  // Function storage: private to this thread's call.
//     uint result = get_work_group_id(&argument);
//   }
//
// This method emits the two statements inside caller() into the kernel being compiled; it does not create a separate
// caller function. On first use, it also declares the imported helper and caches its reference in
// ref_to_imported_function_declaration. glslang supplies the helper body. The GLSL parameter retains value semantics: a
// write through the pointer changes only the temporary argument, not the caller's original input.
Value IRBuilder::call_glsl_u32_to_u32(Value &ref_to_imported_function_declaration,
                                      const char *name,
                                      uint32_t argument_value) {
  // On first use, declare the imported helper and cache its reference in ref_to_imported_function_declaration.
  if (ref_to_imported_function_declaration.id == 0) {
    SType p_uint32_type = get_pointer_type(t_uint32_, spv::StorageClassFunction);
    SType ref_to_function_type_declaration;
    ref_to_function_type_declaration.id = id_counter_++;
    // The function type declaration is stored in global_, and ref_to_function_type_declaration holds a reference to it.
    // A function type declaration is like a function declaration, but is not named.
    ib_.begin(spv::OpTypeFunction).add_seq(ref_to_function_type_declaration, t_uint32_, p_uint32_type).commit(&global_);
    ref_to_imported_function_declaration = new_value(ref_to_function_type_declaration, ValueKind::kFunction);
    decorate(spv::OpDecorate, ref_to_imported_function_declaration, spv::DecorationLinkageAttributes, name,
             spv::LinkageTypeImport);
    ib_.begin(spv::OpFunction)
        .add_seq(/* return_type= */ t_uint32_,
                 /* result_id= */ ref_to_imported_function_declaration,
                 /* function_control= */ 0,
                 /* function_type= */ ref_to_function_type_declaration)
        .commit(&imported_glsl_function_declarations_);
    Value parameter = new_value(p_uint32_type, ValueKind::kVariablePtr);
    ib_.begin(spv::OpFunctionParameter).add_seq(p_uint32_type, parameter).commit(&imported_glsl_function_declarations_);
    ib_.begin(spv::OpFunctionEnd).commit(&imported_glsl_function_declarations_);
  }
  // GLSL passes scalar function arguments through Function-storage pointers.
  Value argument = alloca_variable(t_uint32_);
  store_variable(argument, uint_immediate_number(t_uint32_, argument_value));
  return make_value(spv::OpFunctionCall, t_uint32_, ref_to_imported_function_declaration, argument);
}

Value IRBuilder::alloca_variable(const SType &type) {
  SType ptr_type = get_pointer_type(type, spv::StorageClassFunction);
  Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable).add_seq(ptr_type, ret, spv::StorageClassFunction).commit(&func_header_);
  return ret;
}

Value IRBuilder::alloca_workgroup_array(const SType &arr_type) {
  SType ptr_type = get_pointer_type(arr_type, spv::StorageClassWorkgroup);
  Value ret = new_value(ptr_type, ValueKind::kVariablePtr);
  ib_.begin(spv::OpVariable).add_seq(ptr_type, ret, spv::StorageClassWorkgroup).commit(&global_);
  return ret;
}

Value IRBuilder::load_variable(Value pointer, const SType &res_type) {
  QD_ASSERT(pointer.flag == ValueKind::kVariablePtr || pointer.flag == ValueKind::kStructArrayPtr ||
            pointer.flag == ValueKind::kPhysicalPtr);
  Value ret = new_value(res_type, ValueKind::kNormal);
  if (pointer.flag == ValueKind::kPhysicalPtr) {
    uint32_t alignment = uint32_t(get_primitive_type_size(res_type.dt));
    // Volatile prevents SPIRV-Cross from forwarding the load as an inline
    // pointer dereference.  Without it, SPIRV-Cross may re-read from the
    // physical pointer at each use site, which produces wrong results when
    // the pointed-to memory is modified between the load and the use (e.g.
    // insertion-sort shifting elements in the same array).
    ib_.begin(spv::OpLoad)
        .add_seq(res_type, ret, pointer, spv::MemoryAccessAlignedMask | spv::MemoryAccessVolatileMask, alignment)
        .commit(&function_);
  } else {
    ib_.begin(spv::OpLoad).add_seq(res_type, ret, pointer).commit(&function_);
  }
  return ret;
}

Value IRBuilder::load_variable_volatile(Value pointer, const SType &res_type) {
  // Like `load_variable`, but always emits the `Volatile` `MemoryAccess` mask -- including for the buffer-backed
  // `kStructArrayPtr` path that the default helper does not decorate.  This is the SPIR-V analogue of LLVM's
  // `LoadInst::setVolatile(true)`: SPIRV-Cross propagates `Volatile` into the generated MSL / GLSL as a re-read on
  // every use, and the SPIR-V optimiser is forbidden from forwarding or merging the load with prior reads of the
  // same address.  Required for `qd.volatile_load` spin-wait correctness on Vulkan / Metal.
  QD_ASSERT(pointer.flag == ValueKind::kVariablePtr || pointer.flag == ValueKind::kStructArrayPtr ||
            pointer.flag == ValueKind::kPhysicalPtr);
  Value ret = new_value(res_type, ValueKind::kNormal);
  if (pointer.flag == ValueKind::kPhysicalPtr) {
    // Physical pointers already require the Aligned mask; OR Volatile in alongside it.  Same encoding as the
    // default `load_variable` physical-pointer path.
    uint32_t alignment = uint32_t(get_primitive_type_size(res_type.dt));
    ib_.begin(spv::OpLoad)
        .add_seq(res_type, ret, pointer, spv::MemoryAccessAlignedMask | spv::MemoryAccessVolatileMask, alignment)
        .commit(&function_);
  } else {
    // Logical pointers (variable / struct-array) accept a bare `MemoryAccess` mask without alignment.
    ib_.begin(spv::OpLoad).add_seq(res_type, ret, pointer, spv::MemoryAccessVolatileMask).commit(&function_);
  }
  return ret;
}
void IRBuilder::store_variable(Value pointer, Value value) {
  QD_ASSERT(pointer.flag == ValueKind::kVariablePtr || pointer.flag == ValueKind::kPhysicalPtr);
  QD_ASSERT(value.stype.id == pointer.stype.element_type_id);
  if (pointer.flag == ValueKind::kPhysicalPtr) {
    uint32_t alignment = uint32_t(get_primitive_type_size(value.stype.dt));
    ib_.begin(spv::OpStore).add_seq(pointer, value, spv::MemoryAccessAlignedMask, alignment).commit(&function_);
  } else {
    ib_.begin(spv::OpStore).add_seq(pointer, value).commit(&function_);
  }
}

void IRBuilder::register_value(std::string name, Value value) {
  auto it = value_name_tbl_.find(name);
  if (it != value_name_tbl_.end() && it->second.flag != ValueKind::kConstant) {
    QD_ERROR("{} already exists.", name);
  }
  this->debug_name(spv::OpName, value, fmt::format("{}_{}", name, value.stype.dt.to_string()));  // Debug info
  value_name_tbl_[name] = value;
}

Value IRBuilder::query_value(std::string name) const {
  auto it = value_name_tbl_.find(name);
  if (it != value_name_tbl_.end()) {
    return it->second;
  }
  QD_ERROR("Value \"{}\" does not yet exist.", name);
}

bool IRBuilder::check_value_existence(const std::string &name) const {
  return value_name_tbl_.find(name) != value_name_tbl_.end();
}

Value IRBuilder::get_const(const SType &dtype, const uint64_t *pvalue, bool cache) {
  auto key = std::make_pair(dtype.id, pvalue[0]);
  if (cache) {
    auto it = const_tbl_.find(key);
    if (it != const_tbl_.end()) {
      return it->second;
    }
  }

  QD_WARN_IF(dtype.flag != TypeKind::kPrimitive, "Trying to get const with dtype.flag={} , .dt={}", int(dtype.flag),
             dtype.dt.to_string());
  Value ret = new_value(dtype, ValueKind::kConstant);
  if (dtype.dt->is_primitive(PrimitiveTypeID::u1)) {
    // bool type
    if (*pvalue) {
      ib_.begin(spv::OpConstantTrue).add_seq(dtype, ret);
    } else {
      ib_.begin(spv::OpConstantFalse).add_seq(dtype, ret);
    }
  } else {
    // Integral/floating-point types.
    ib_.begin(spv::OpConstant).add_seq(dtype, ret);
    uint64_t mask = 0xFFFFFFFFUL;
    ib_.add(static_cast<uint32_t>(pvalue[0] & mask));
    if (data_type_bits(dtype.dt) > 32) {
      if (is_integral(dtype.dt)) {
        int64_t sign_mask = 0xFFFFFFFFL;
        const int64_t *sign_ptr = reinterpret_cast<const int64_t *>(pvalue);
        ib_.add(static_cast<uint32_t>((sign_ptr[0] >> 32L) & sign_mask));
      } else {
        ib_.add(static_cast<uint32_t>((pvalue[0] >> 32UL) & mask));
      }
    }
  }

  ib_.commit(&global_);
  if (cache) {
    const_tbl_[key] = ret;
  }
  return ret;
}

SType IRBuilder::declare_primitive_type(DataType dt) {
  SType t;
  t.id = id_counter_++;
  t.dt = dt;
  t.flag = TypeKind::kPrimitive;

  dt.set_is_pointer(false);
  if (dt->is_primitive(PrimitiveTypeID::u1))
    ib_.begin(spv::OpTypeBool).add(t).commit(&global_);
  else if (is_real(dt))
    ib_.begin(spv::OpTypeFloat).add_seq(t, data_type_bits(dt)).commit(&global_);
  else if (is_integral(dt))
    ib_.begin(spv::OpTypeInt).add_seq(t, data_type_bits(dt), static_cast<int>(is_signed(dt))).commit(&global_);
  else {
    QD_ERROR("Type {} not supported.", dt->to_string());
  }

  return t;
}

Value IRBuilder::make_access_chain(const SType &out_type, Value base, const std::vector<int> &indices) {
  Value ret = new_value(out_type, ValueKind::kVariablePtr);
  std::vector<Value> index_values;
  for (auto &ind : indices) {
    index_values.push_back(int_immediate_number(t_int32_, ind));
  }
  ib_.begin(spv::OpAccessChain).add_seq(out_type, ret, base);
  for (auto &ind : index_values) {
    ib_.add(ind);
  }
  ib_.commit(&function_);
  return ret;
}

}  // namespace spirv
}  // namespace quadrants::lang
