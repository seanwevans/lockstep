; ModuleID = "lockstep"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%"struct.Reading" = type {i32, float, i1}
%"struct.Lockstep_Arena" = type {[4096 x i32], [4096 x float], [4096 x i1], [4096 x i32], [4096 x float], [4096 x i1], [4096 x i32], [4096 x float], [4096 x i1], [4096 x float]}
declare float @"pure_step"(float %"edge", float %"x")

declare float @"pure_mix"(float %"a", float %"b", float %"t")

declare float @"pure_clamp"(float %"x", float %"min_value", float %"max_value")

declare float @"pure_max"(float %"x", float %"y")

declare float @"pure_min"(float %"x", float %"y")

declare float @"pure_abs"(float %"x")

declare float @"pure_sign"(float %"x")

declare float @"pure_smoothstep"(float %"edge0", float %"edge1", float %"x")

define void @"shader_Aggregate"(%"struct.Reading" %"src", %"struct.Reading"* %"dst", float* %"total")
{
entry:
  %"src.1" = alloca %"struct.Reading"
  store %"struct.Reading" %"src", %"struct.Reading"* %"src.1"
  %"dst.1" = alloca %"struct.Reading"*
  store %"struct.Reading"* %"dst", %"struct.Reading"** %"dst.1"
  %"total.1" = alloca float*
  store float* %"total", float** %"total.1"
  %"total_val" = load float*, float** %"total.1"
  %"total_ref" = load float, float* %"total_val"
  %"src_val" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"value_field" = extractvalue %"struct.Reading" %"src_val", 1
  %".8" = fadd float %"total_ref", %"value_field"
  %"total_ptr" = load float*, float** %"total.1"
  store float %".8", float* %"total_ptr"
  %"src_val.1" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"sensorId_field" = extractvalue %"struct.Reading" %"src_val.1", 0
  %"dst_ptr" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  %"dst_ref" = load %"struct.Reading", %"struct.Reading"* %"dst_ptr"
  %"set_sensorId" = insertvalue %"struct.Reading" %"dst_ref", i32 %"sensorId_field", 0
  %"dst_ptr.1" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  store %"struct.Reading" %"set_sensorId", %"struct.Reading"* %"dst_ptr.1"
  %"src_val.2" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"value_field.1" = extractvalue %"struct.Reading" %"src_val.2", 1
  %"dst_ptr.2" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  %"dst_ref.1" = load %"struct.Reading", %"struct.Reading"* %"dst_ptr.2"
  %"set_value" = insertvalue %"struct.Reading" %"dst_ref.1", float %"value_field.1", 1
  %"dst_ptr.3" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  store %"struct.Reading" %"set_value", %"struct.Reading"* %"dst_ptr.3"
  %"src_val.3" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"valid_field" = extractvalue %"struct.Reading" %"src_val.3", 2
  %"dst_ptr.4" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  %"dst_ref.2" = load %"struct.Reading", %"struct.Reading"* %"dst_ptr.4"
  %"set_valid" = insertvalue %"struct.Reading" %"dst_ref.2", i1 %"valid_field", 2
  %"dst_ptr.5" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  store %"struct.Reading" %"set_valid", %"struct.Reading"* %"dst_ptr.5"
  ret void
}

define i1 @"filter_DropInvalid"(%"struct.Reading" %"src", %"struct.Reading"* %"dst")
{
entry:
  %"src.1" = alloca %"struct.Reading"
  store %"struct.Reading" %"src", %"struct.Reading"* %"src.1"
  %"dst.1" = alloca %"struct.Reading"*
  store %"struct.Reading"* %"dst", %"struct.Reading"** %"dst.1"
  %"src_val" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"sensorId_field" = extractvalue %"struct.Reading" %"src_val", 0
  %"dst_ptr" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  %"dst_ref" = load %"struct.Reading", %"struct.Reading"* %"dst_ptr"
  %"set_sensorId" = insertvalue %"struct.Reading" %"dst_ref", i32 %"sensorId_field", 0
  %"dst_ptr.1" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  store %"struct.Reading" %"set_sensorId", %"struct.Reading"* %"dst_ptr.1"
  %"src_val.1" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"value_field" = extractvalue %"struct.Reading" %"src_val.1", 1
  %"dst_ptr.2" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  %"dst_ref.1" = load %"struct.Reading", %"struct.Reading"* %"dst_ptr.2"
  %"set_value" = insertvalue %"struct.Reading" %"dst_ref.1", float %"value_field", 1
  %"dst_ptr.3" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  store %"struct.Reading" %"set_value", %"struct.Reading"* %"dst_ptr.3"
  %"src_val.2" = load %"struct.Reading", %"struct.Reading"* %"src.1"
  %"valid_field" = extractvalue %"struct.Reading" %"src_val.2", 2
  %"dst_ptr.4" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  %"dst_ref.2" = load %"struct.Reading", %"struct.Reading"* %"dst_ptr.4"
  %"set_valid" = insertvalue %"struct.Reading" %"dst_ref.2", i1 %"valid_field", 2
  %"dst_ptr.5" = load %"struct.Reading"*, %"struct.Reading"** %"dst.1"
  store %"struct.Reading" %"set_valid", %"struct.Reading"* %"dst_ptr.5"
  ret i1 1
}

define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* noalias nocapture %"arena")
{
entry:
  %"fused_0_idx" = alloca i32
  store i32 0, i32* %"fused_0_idx"
  %"fused_0_grandTotal_vec" = alloca <8 x float>
  store <8 x float> <float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0>, <8 x float>* %"fused_0_grandTotal_vec"
  %"fused_0_grandTotal_tail" = alloca float
  store float              0x0, float* %"fused_0_grandTotal_tail"
  br label %"fused_0_cond"
fused_0_cond:
  %"fused_idx" = load i32, i32* %"fused_0_idx"
  %"fused_vector_active" = icmp slt i32 %"fused_idx", 4096
  br i1 %"fused_vector_active", label %"fused_0_body", label %"fused_0_exit"
fused_0_body:
  %"stream_readingsRaw_byte_index" = mul i32 %"fused_idx", 4
  %"stream_readingsRaw_byte_offset" = add i32 0, %"stream_readingsRaw_byte_index"
  %"stream_readingsRaw_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsRaw_sensorId_byte_ptr" = getelementptr i8, i8* %"stream_readingsRaw_arena_bytes", i32 %"stream_readingsRaw_byte_offset"
  %".8" = bitcast i8* %"stream_readingsRaw_sensorId_byte_ptr" to i32*
  %"fused_readingsRaw_sensorId_vector_ptr" = bitcast i32* %".8" to <8 x i32>*
  %"fused_readingsRaw_sensorId_vector" = load <8 x i32>, <8 x i32>* %"fused_readingsRaw_sensorId_vector_ptr", align 1
  %"stream_readingsRaw_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_readingsRaw_byte_offset.1" = add i32 16384, %"stream_readingsRaw_byte_index.1"
  %"stream_readingsRaw_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsRaw_value_byte_ptr" = getelementptr i8, i8* %"stream_readingsRaw_arena_bytes.1", i32 %"stream_readingsRaw_byte_offset.1"
  %".9" = bitcast i8* %"stream_readingsRaw_value_byte_ptr" to float*
  %"fused_readingsRaw_value_vector_ptr" = bitcast float* %".9" to <8 x float>*
  %"fused_readingsRaw_value_vector" = load <8 x float>, <8 x float>* %"fused_readingsRaw_value_vector_ptr", align 1
  %"stream_readingsRaw_byte_index.2" = mul i32 %"fused_idx", 1
  %"stream_readingsRaw_byte_offset.2" = add i32 32768, %"stream_readingsRaw_byte_index.2"
  %"stream_readingsRaw_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsRaw_valid_byte_ptr" = getelementptr i8, i8* %"stream_readingsRaw_arena_bytes.2", i32 %"stream_readingsRaw_byte_offset.2"
  %"fused_readingsRaw_valid_vector_ptr" = bitcast i8* %"stream_readingsRaw_valid_byte_ptr" to <8 x i8>*
  %"fused_readingsRaw_valid_vector" = load <8 x i8>, <8 x i8>* %"fused_readingsRaw_valid_vector_ptr", align 1
  %"fused_readingsRaw_valid_trunc" = trunc <8 x i8> %"fused_readingsRaw_valid_vector" to <8 x i1>
  %"stream_readingsValid_byte_index" = mul i32 %"fused_idx", 4
  %"stream_readingsValid_byte_offset" = add i32 36864, %"stream_readingsValid_byte_index"
  %"stream_readingsValid_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_sensorId_byte_ptr" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes", i32 %"stream_readingsValid_byte_offset"
  %".10" = bitcast i8* %"stream_readingsValid_sensorId_byte_ptr" to i32*
  %"fused_readingsValid_sensorId_vector_ptr" = bitcast i32* %".10" to <8 x i32>*
  %"fused_readingsValid_sensorId_vector" = load <8 x i32>, <8 x i32>* %"fused_readingsValid_sensorId_vector_ptr", align 1
  %"stream_readingsValid_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_readingsValid_byte_offset.1" = add i32 53248, %"stream_readingsValid_byte_index.1"
  %"stream_readingsValid_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_value_byte_ptr" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.1", i32 %"stream_readingsValid_byte_offset.1"
  %".11" = bitcast i8* %"stream_readingsValid_value_byte_ptr" to float*
  %"fused_readingsValid_value_vector_ptr" = bitcast float* %".11" to <8 x float>*
  %"fused_readingsValid_value_vector" = load <8 x float>, <8 x float>* %"fused_readingsValid_value_vector_ptr", align 1
  %"stream_readingsValid_byte_index.2" = mul i32 %"fused_idx", 1
  %"stream_readingsValid_byte_offset.2" = add i32 69632, %"stream_readingsValid_byte_index.2"
  %"stream_readingsValid_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_valid_byte_ptr" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.2", i32 %"stream_readingsValid_byte_offset.2"
  %"fused_readingsValid_valid_vector_ptr" = bitcast i8* %"stream_readingsValid_valid_byte_ptr" to <8 x i8>*
  %"fused_readingsValid_valid_vector" = load <8 x i8>, <8 x i8>* %"fused_readingsValid_valid_vector_ptr", align 1
  %"fused_readingsValid_valid_trunc" = trunc <8 x i8> %"fused_readingsValid_valid_vector" to <8 x i1>
  %"stream_readingsAggregated_byte_index" = mul i32 %"fused_idx", 4
  %"stream_readingsAggregated_byte_offset" = add i32 73728, %"stream_readingsAggregated_byte_index"
  %"stream_readingsAggregated_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_sensorId_byte_ptr" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes", i32 %"stream_readingsAggregated_byte_offset"
  %".12" = bitcast i8* %"stream_readingsAggregated_sensorId_byte_ptr" to i32*
  %"fused_readingsAggregated_sensorId_vector_ptr" = bitcast i32* %".12" to <8 x i32>*
  %"fused_readingsAggregated_sensorId_vector" = load <8 x i32>, <8 x i32>* %"fused_readingsAggregated_sensorId_vector_ptr", align 1
  %"stream_readingsAggregated_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_readingsAggregated_byte_offset.1" = add i32 90112, %"stream_readingsAggregated_byte_index.1"
  %"stream_readingsAggregated_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_value_byte_ptr" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.1", i32 %"stream_readingsAggregated_byte_offset.1"
  %".13" = bitcast i8* %"stream_readingsAggregated_value_byte_ptr" to float*
  %"fused_readingsAggregated_value_vector_ptr" = bitcast float* %".13" to <8 x float>*
  %"fused_readingsAggregated_value_vector" = load <8 x float>, <8 x float>* %"fused_readingsAggregated_value_vector_ptr", align 1
  %"stream_readingsAggregated_byte_index.2" = mul i32 %"fused_idx", 1
  %"stream_readingsAggregated_byte_offset.2" = add i32 106496, %"stream_readingsAggregated_byte_index.2"
  %"stream_readingsAggregated_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_valid_byte_ptr" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.2", i32 %"stream_readingsAggregated_byte_offset.2"
  %"fused_readingsAggregated_valid_vector_ptr" = bitcast i8* %"stream_readingsAggregated_valid_byte_ptr" to <8 x i8>*
  %"fused_readingsAggregated_valid_vector" = load <8 x i8>, <8 x i8>* %"fused_readingsAggregated_valid_vector_ptr", align 1
  %"fused_readingsAggregated_valid_trunc" = trunc <8 x i8> %"fused_readingsAggregated_valid_vector" to <8 x i1>
  %"fused_math" = fadd <8 x float> <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, %"fused_readingsRaw_value_vector"
  %"stream_readingsAggregated_byte_index.3" = mul i32 %"fused_idx", 4
  %"stream_readingsAggregated_byte_offset.3" = add i32 73728, %"stream_readingsAggregated_byte_index.3"
  %"stream_readingsAggregated_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_sensorId_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.3", i32 %"stream_readingsAggregated_byte_offset.3"
  %".14" = bitcast i8* %"stream_readingsAggregated_sensorId_byte_ptr.1" to i32*
  %"fused_readingsAggregated_sensorId_vector_ptr.1" = bitcast i32* %".14" to <8 x i32>*
  store <8 x i32> %"fused_readingsRaw_sensorId_vector", <8 x i32>* %"fused_readingsAggregated_sensorId_vector_ptr.1", align 1
  %"stream_readingsAggregated_byte_index.4" = mul i32 %"fused_idx", 4
  %"stream_readingsAggregated_byte_offset.4" = add i32 90112, %"stream_readingsAggregated_byte_index.4"
  %"stream_readingsAggregated_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_value_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.4", i32 %"stream_readingsAggregated_byte_offset.4"
  %".16" = bitcast i8* %"stream_readingsAggregated_value_byte_ptr.1" to float*
  %"fused_readingsAggregated_value_vector_ptr.1" = bitcast float* %".16" to <8 x float>*
  store <8 x float> %"fused_readingsRaw_value_vector", <8 x float>* %"fused_readingsAggregated_value_vector_ptr.1", align 1
  %"stream_readingsAggregated_byte_index.5" = mul i32 %"fused_idx", 1
  %"stream_readingsAggregated_byte_offset.5" = add i32 106496, %"stream_readingsAggregated_byte_index.5"
  %"stream_readingsAggregated_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_valid_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.5", i32 %"stream_readingsAggregated_byte_offset.5"
  %"fused_readingsAggregated_valid_vector_ptr.1" = bitcast i8* %"stream_readingsAggregated_valid_byte_ptr.1" to <8 x i8>*
  %"fused_store_readingsAggregated_valid_zext" = zext <8 x i1> %"fused_readingsRaw_valid_trunc" to <8 x i8>
  store <8 x i8> %"fused_store_readingsAggregated_valid_zext", <8 x i8>* %"fused_readingsAggregated_valid_vector_ptr.1", align 1
  %"fused_carry_cur" = load <8 x float>, <8 x float>* %"fused_0_grandTotal_vec"
  %"fused_carry_next" = fadd fast <8 x float> %"fused_carry_cur", %"fused_math"
  store <8 x float> %"fused_carry_next", <8 x float>* %"fused_0_grandTotal_vec"
  %"fused_idx_next" = add i32 %"fused_idx", 8
  store i32 %"fused_idx_next", i32* %"fused_0_idx"
  br label %"fused_0_cond"
fused_0_exit:
  %"fused_carry_final_vec" = load <8 x float>, <8 x float>* %"fused_0_grandTotal_vec"
  %"fused_carry_reduce" = call fast float @"llvm.vector.reduce.fadd.v8f32"(float              0x0, <8 x float> %"fused_carry_final_vec")
  %"fused_carry_final_tail" = load float, float* %"fused_0_grandTotal_tail"
  %"fused_carry_final" = fadd fast float %"fused_carry_reduce", %"fused_carry_final_tail"
  ret void
}

declare float @"llvm.vector.reduce.fadd.v8f32"(float %".1", <8 x float> %".2")
