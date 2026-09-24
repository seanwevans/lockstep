; ModuleID = "lockstep"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%"struct.Sample" = type {i32, float, i1}
%"struct.Lockstep_Arena" = type {[20 x i32], [20 x float], [20 x i1], [20 x i32], [20 x float], [20 x i1], [20 x i32], [20 x float], [20 x i1], [20 x float], float, i32, i32}
declare float @"pure_step"(float %"edge", float %"x")

declare float @"pure_mix"(float %"a", float %"b", float %"t")

declare float @"pure_clamp"(float %"x", float %"min_value", float %"max_value")

declare float @"pure_max"(float %"x", float %"y")

declare float @"pure_min"(float %"x", float %"y")

declare float @"pure_abs"(float %"x")

declare float @"pure_sign"(float %"x")

declare float @"pure_smoothstep"(float %"edge0", float %"edge1", float %"x")

define void @"shader_Scale"(%"struct.Sample" %"src", %"struct.Sample"* %"dst", float* %"total")
{
entry:
  %"src.1" = alloca %"struct.Sample"
  store %"struct.Sample" %"src", %"struct.Sample"* %"src.1"
  %"dst.1" = alloca %"struct.Sample"*
  store %"struct.Sample"* %"dst", %"struct.Sample"** %"dst.1"
  %"total.1" = alloca float*
  store float* %"total", float** %"total.1"
  %"src_val" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"id_field" = extractvalue %"struct.Sample" %"src_val", 0
  %"dst_ptr" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  %"dst_ref" = load %"struct.Sample", %"struct.Sample"* %"dst_ptr"
  %"set_id" = insertvalue %"struct.Sample" %"dst_ref", i32 %"id_field", 0
  %"dst_ptr.1" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  store %"struct.Sample" %"set_id", %"struct.Sample"* %"dst_ptr.1"
  %"src_val.1" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"value_field" = extractvalue %"struct.Sample" %"src_val.1", 1
  %".9" = fmul float %"value_field", 0x4000000000000000
  %"dst_ptr.2" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  %"dst_ref.1" = load %"struct.Sample", %"struct.Sample"* %"dst_ptr.2"
  %"set_value" = insertvalue %"struct.Sample" %"dst_ref.1", float %".9", 1
  %"dst_ptr.3" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  store %"struct.Sample" %"set_value", %"struct.Sample"* %"dst_ptr.3"
  %"src_val.2" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"flagged_field" = extractvalue %"struct.Sample" %"src_val.2", 2
  %"dst_ptr.4" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  %"dst_ref.2" = load %"struct.Sample", %"struct.Sample"* %"dst_ptr.4"
  %"set_flagged" = insertvalue %"struct.Sample" %"dst_ref.2", i1 %"flagged_field", 2
  %"dst_ptr.5" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  store %"struct.Sample" %"set_flagged", %"struct.Sample"* %"dst_ptr.5"
  %"total_val" = load float*, float** %"total.1"
  %"total_ref" = load float, float* %"total_val"
  %"src_val.3" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"value_field.1" = extractvalue %"struct.Sample" %"src_val.3", 1
  %".12" = fadd float %"total_ref", %"value_field.1"
  %"total_ptr" = load float*, float** %"total.1"
  store float %".12", float* %"total_ptr"
  ret void
}

define i1 @"filter_DropFlagged"(%"struct.Sample" %"src", %"struct.Sample"* %"dst")
{
entry:
  %"src.1" = alloca %"struct.Sample"
  store %"struct.Sample" %"src", %"struct.Sample"* %"src.1"
  %"dst.1" = alloca %"struct.Sample"*
  store %"struct.Sample"* %"dst", %"struct.Sample"** %"dst.1"
  %"src_val" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"id_field" = extractvalue %"struct.Sample" %"src_val", 0
  %"dst_ptr" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  %"dst_ref" = load %"struct.Sample", %"struct.Sample"* %"dst_ptr"
  %"set_id" = insertvalue %"struct.Sample" %"dst_ref", i32 %"id_field", 0
  %"dst_ptr.1" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  store %"struct.Sample" %"set_id", %"struct.Sample"* %"dst_ptr.1"
  %"src_val.1" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"value_field" = extractvalue %"struct.Sample" %"src_val.1", 1
  %"dst_ptr.2" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  %"dst_ref.1" = load %"struct.Sample", %"struct.Sample"* %"dst_ptr.2"
  %"set_value" = insertvalue %"struct.Sample" %"dst_ref.1", float %"value_field", 1
  %"dst_ptr.3" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  store %"struct.Sample" %"set_value", %"struct.Sample"* %"dst_ptr.3"
  %"src_val.2" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"flagged_field" = extractvalue %"struct.Sample" %"src_val.2", 2
  %"dst_ptr.4" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  %"dst_ref.2" = load %"struct.Sample", %"struct.Sample"* %"dst_ptr.4"
  %"set_flagged" = insertvalue %"struct.Sample" %"dst_ref.2", i1 %"flagged_field", 2
  %"dst_ptr.5" = load %"struct.Sample"*, %"struct.Sample"** %"dst.1"
  store %"struct.Sample" %"set_flagged", %"struct.Sample"* %"dst_ptr.5"
  %"src_val.3" = load %"struct.Sample", %"struct.Sample"* %"src.1"
  %"flagged_field.1" = extractvalue %"struct.Sample" %"src_val.3", 2
  %".9" = xor i1 %"flagged_field.1", -1
  ret i1 %".9"
}

define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* noalias nocapture %"arena")
{
entry:
  %"fused_0_idx" = alloca i32
  store i32 0, i32* %"fused_0_idx"
  %"fused_0_write_idx" = alloca i32
  store i32 0, i32* %"fused_0_write_idx"
  %"fused_0_keptTotal_vec" = alloca <8 x float>
  store <8 x float> <float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0>, <8 x float>* %"fused_0_keptTotal_vec"
  %"fused_0_keptTotal_tail" = alloca float
  store float              0x0, float* %"fused_0_keptTotal_tail"
  br label %"fused_0_cond"
fused_0_cond:
  %"fused_idx" = load i32, i32* %"fused_0_idx"
  %"fused_vector_active" = icmp slt i32 %"fused_idx", 16
  br i1 %"fused_vector_active", label %"fused_0_body", label %"fused_0_exit"
fused_0_body:
  %"fused_write_idx" = load i32, i32* %"fused_0_write_idx"
  %"stream_samplesRaw_byte_index" = mul i32 %"fused_idx", 4
  %"stream_samplesRaw_byte_offset" = add i32 0, %"stream_samplesRaw_byte_index"
  %"stream_samplesRaw_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesRaw_id_byte_ptr" = getelementptr i8, i8* %"stream_samplesRaw_arena_bytes", i32 %"stream_samplesRaw_byte_offset"
  %".9" = bitcast i8* %"stream_samplesRaw_id_byte_ptr" to i32*
  %"fused_samplesRaw_id_vector_ptr" = bitcast i32* %".9" to <8 x i32>*
  %"fused_samplesRaw_id_vector" = load <8 x i32>, <8 x i32>* %"fused_samplesRaw_id_vector_ptr", align 1
  %"stream_samplesRaw_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_samplesRaw_byte_offset.1" = add i32 80, %"stream_samplesRaw_byte_index.1"
  %"stream_samplesRaw_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesRaw_value_byte_ptr" = getelementptr i8, i8* %"stream_samplesRaw_arena_bytes.1", i32 %"stream_samplesRaw_byte_offset.1"
  %".10" = bitcast i8* %"stream_samplesRaw_value_byte_ptr" to float*
  %"fused_samplesRaw_value_vector_ptr" = bitcast float* %".10" to <8 x float>*
  %"fused_samplesRaw_value_vector" = load <8 x float>, <8 x float>* %"fused_samplesRaw_value_vector_ptr", align 1
  %"stream_samplesRaw_byte_index.2" = mul i32 %"fused_idx", 1
  %"stream_samplesRaw_byte_offset.2" = add i32 160, %"stream_samplesRaw_byte_index.2"
  %"stream_samplesRaw_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesRaw_flagged_byte_ptr" = getelementptr i8, i8* %"stream_samplesRaw_arena_bytes.2", i32 %"stream_samplesRaw_byte_offset.2"
  %"fused_samplesRaw_flagged_vector_ptr" = bitcast i8* %"stream_samplesRaw_flagged_byte_ptr" to <8 x i8>*
  %"fused_samplesRaw_flagged_vector" = load <8 x i8>, <8 x i8>* %"fused_samplesRaw_flagged_vector_ptr", align 1
  %"fused_samplesRaw_flagged_trunc" = trunc <8 x i8> %"fused_samplesRaw_flagged_vector" to <8 x i1>
  %"stream_samplesKept_byte_index" = mul i32 %"fused_idx", 4
  %"stream_samplesKept_byte_offset" = add i32 180, %"stream_samplesKept_byte_index"
  %"stream_samplesKept_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesKept_id_byte_ptr" = getelementptr i8, i8* %"stream_samplesKept_arena_bytes", i32 %"stream_samplesKept_byte_offset"
  %".11" = bitcast i8* %"stream_samplesKept_id_byte_ptr" to i32*
  %"fused_samplesKept_id_vector_ptr" = bitcast i32* %".11" to <8 x i32>*
  %"fused_samplesKept_id_vector" = load <8 x i32>, <8 x i32>* %"fused_samplesKept_id_vector_ptr", align 1
  %"stream_samplesKept_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_samplesKept_byte_offset.1" = add i32 260, %"stream_samplesKept_byte_index.1"
  %"stream_samplesKept_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesKept_value_byte_ptr" = getelementptr i8, i8* %"stream_samplesKept_arena_bytes.1", i32 %"stream_samplesKept_byte_offset.1"
  %".12" = bitcast i8* %"stream_samplesKept_value_byte_ptr" to float*
  %"fused_samplesKept_value_vector_ptr" = bitcast float* %".12" to <8 x float>*
  %"fused_samplesKept_value_vector" = load <8 x float>, <8 x float>* %"fused_samplesKept_value_vector_ptr", align 1
  %"stream_samplesKept_byte_index.2" = mul i32 %"fused_idx", 1
  %"stream_samplesKept_byte_offset.2" = add i32 340, %"stream_samplesKept_byte_index.2"
  %"stream_samplesKept_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesKept_flagged_byte_ptr" = getelementptr i8, i8* %"stream_samplesKept_arena_bytes.2", i32 %"stream_samplesKept_byte_offset.2"
  %"fused_samplesKept_flagged_vector_ptr" = bitcast i8* %"stream_samplesKept_flagged_byte_ptr" to <8 x i8>*
  %"fused_samplesKept_flagged_vector" = load <8 x i8>, <8 x i8>* %"fused_samplesKept_flagged_vector_ptr", align 1
  %"fused_samplesKept_flagged_trunc" = trunc <8 x i8> %"fused_samplesKept_flagged_vector" to <8 x i1>
  %"fused_not" = xor <8 x i1> %"fused_samplesRaw_flagged_trunc", <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %"stream_samplesScaled_byte_index" = mul i32 %"fused_write_idx", 4
  %"stream_samplesScaled_byte_offset" = add i32 360, %"stream_samplesScaled_byte_index"
  %"stream_samplesScaled_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_id_byte_ptr" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes", i32 %"stream_samplesScaled_byte_offset"
  %".13" = bitcast i8* %"stream_samplesScaled_id_byte_ptr" to i32*
  %"fused_samplesScaled_id_vector_ptr" = bitcast i32* %".13" to <8 x i32>*
  %"fused_samplesScaled_id_vector" = load <8 x i32>, <8 x i32>* %"fused_samplesScaled_id_vector_ptr", align 1
  %"stream_samplesScaled_byte_index.1" = mul i32 %"fused_write_idx", 4
  %"stream_samplesScaled_byte_offset.1" = add i32 440, %"stream_samplesScaled_byte_index.1"
  %"stream_samplesScaled_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_value_byte_ptr" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.1", i32 %"stream_samplesScaled_byte_offset.1"
  %".14" = bitcast i8* %"stream_samplesScaled_value_byte_ptr" to float*
  %"fused_samplesScaled_value_vector_ptr" = bitcast float* %".14" to <8 x float>*
  %"fused_samplesScaled_value_vector" = load <8 x float>, <8 x float>* %"fused_samplesScaled_value_vector_ptr", align 1
  %"stream_samplesScaled_byte_index.2" = mul i32 %"fused_write_idx", 1
  %"stream_samplesScaled_byte_offset.2" = add i32 520, %"stream_samplesScaled_byte_index.2"
  %"stream_samplesScaled_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_flagged_byte_ptr" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.2", i32 %"stream_samplesScaled_byte_offset.2"
  %"fused_samplesScaled_flagged_vector_ptr" = bitcast i8* %"stream_samplesScaled_flagged_byte_ptr" to <8 x i8>*
  %"fused_samplesScaled_flagged_vector" = load <8 x i8>, <8 x i8>* %"fused_samplesScaled_flagged_vector_ptr", align 1
  %"fused_samplesScaled_flagged_trunc" = trunc <8 x i8> %"fused_samplesScaled_flagged_vector" to <8 x i1>
  %".15" = insertelement <8 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef>, float 0x4000000000000000, i32 0
  %"fused_splat" = shufflevector <8 x float> %".15", <8 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %"fused_math" = fmul <8 x float> %"fused_samplesRaw_value_vector", %"fused_splat"
  %"fused_math.1" = fadd <8 x float> <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, %"fused_samplesRaw_value_vector"
  %"stream_samplesScaled_byte_index.3" = mul i32 %"fused_write_idx", 4
  %"stream_samplesScaled_byte_offset.3" = add i32 360, %"stream_samplesScaled_byte_index.3"
  %"stream_samplesScaled_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_id_byte_ptr.1" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.3", i32 %"stream_samplesScaled_byte_offset.3"
  %".16" = bitcast i8* %"stream_samplesScaled_id_byte_ptr.1" to i32*
  call void @"llvm.masked.compressstore.v8i32"(<8 x i32> %"fused_samplesRaw_id_vector", i32* %".16", <8 x i1> %"fused_not")
  %"stream_samplesScaled_byte_index.4" = mul i32 %"fused_write_idx", 4
  %"stream_samplesScaled_byte_offset.4" = add i32 440, %"stream_samplesScaled_byte_index.4"
  %"stream_samplesScaled_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_value_byte_ptr.1" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.4", i32 %"stream_samplesScaled_byte_offset.4"
  %".18" = bitcast i8* %"stream_samplesScaled_value_byte_ptr.1" to float*
  call void @"llvm.masked.compressstore.v8f32"(<8 x float> %"fused_math", float* %".18", <8 x i1> %"fused_not")
  %"fused_compress_samplesScaled_zext" = zext <8 x i1> %"fused_samplesRaw_flagged_trunc" to <8 x i8>
  %"stream_samplesScaled_byte_index.5" = mul i32 %"fused_write_idx", 1
  %"stream_samplesScaled_byte_offset.5" = add i32 520, %"stream_samplesScaled_byte_index.5"
  %"stream_samplesScaled_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_flagged_byte_ptr.1" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.5", i32 %"stream_samplesScaled_byte_offset.5"
  call void @"llvm.masked.compressstore.v8i8"(<8 x i8> %"fused_compress_samplesScaled_zext", i8* %"stream_samplesScaled_flagged_byte_ptr.1", <8 x i1> %"fused_not")
  %"fused_kept_lanes" = zext <8 x i1> %"fused_not" to <8 x i32>
  %"fused_kept" = call i32 @"llvm.vector.reduce.add.v8i32"(<8 x i32> %"fused_kept_lanes")
  %"fused_write_next" = add i32 %"fused_write_idx", %"fused_kept"
  store i32 %"fused_write_next", i32* %"fused_0_write_idx"
  %"fused_carry_masked" = select  <8 x i1> %"fused_not", <8 x float> %"fused_math.1", <8 x float> <float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0>
  %"fused_carry_cur" = load <8 x float>, <8 x float>* %"fused_0_keptTotal_vec"
  %"fused_carry_next" = fadd fast <8 x float> %"fused_carry_cur", %"fused_carry_masked"
  store <8 x float> %"fused_carry_next", <8 x float>* %"fused_0_keptTotal_vec"
  %"fused_idx_next" = add i32 %"fused_idx", 8
  store i32 %"fused_idx_next", i32* %"fused_0_idx"
  br label %"fused_0_cond"
fused_0_exit:
  %"fused_0_tail_idx" = alloca i32
  store i32 16, i32* %"fused_0_tail_idx"
  br label %"fused_0_tail_cond"
fused_0_tail_cond:
  %"fused_tail_idx" = load i32, i32* %"fused_0_tail_idx"
  %"fused_tail_active" = icmp slt i32 %"fused_tail_idx", 20
  br i1 %"fused_tail_active", label %"fused_0_tail_body", label %"fused_0_tail_exit"
fused_0_tail_body:
  %"fused_samplesKept_tail_slot" = alloca %"struct.Sample"
  %"route_clamp_lo_cmp" = icmp sgt i32 %"fused_tail_idx", 0
  %"route_clamp_lo" = select  i1 %"route_clamp_lo_cmp", i32 %"fused_tail_idx", i32 0
  %"route_clamp_hi_cmp" = icmp slt i32 %"route_clamp_lo", 19
  %"route_clamp" = select  i1 %"route_clamp_hi_cmp", i32 %"route_clamp_lo", i32 19
  %"stream_samplesRaw_byte_index.3" = mul i32 %"route_clamp", 4
  %"stream_samplesRaw_byte_offset.3" = add i32 0, %"stream_samplesRaw_byte_index.3"
  %"stream_samplesRaw_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesRaw_id_byte_ptr.1" = getelementptr i8, i8* %"stream_samplesRaw_arena_bytes.3", i32 %"stream_samplesRaw_byte_offset.3"
  %".28" = bitcast i8* %"stream_samplesRaw_id_byte_ptr.1" to i32*
  %"stream_samplesRaw_val" = load i32, i32* %".28"
  %".29" = insertvalue %"struct.Sample" undef, i32 %"stream_samplesRaw_val", 0
  %"stream_samplesRaw_byte_index.4" = mul i32 %"route_clamp", 4
  %"stream_samplesRaw_byte_offset.4" = add i32 80, %"stream_samplesRaw_byte_index.4"
  %"stream_samplesRaw_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesRaw_value_byte_ptr.1" = getelementptr i8, i8* %"stream_samplesRaw_arena_bytes.4", i32 %"stream_samplesRaw_byte_offset.4"
  %".30" = bitcast i8* %"stream_samplesRaw_value_byte_ptr.1" to float*
  %"stream_samplesRaw_val.1" = load float, float* %".30"
  %".31" = insertvalue %"struct.Sample" %".29", float %"stream_samplesRaw_val.1", 1
  %"stream_samplesRaw_byte_index.5" = mul i32 %"route_clamp", 1
  %"stream_samplesRaw_byte_offset.5" = add i32 160, %"stream_samplesRaw_byte_index.5"
  %"stream_samplesRaw_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesRaw_flagged_byte_ptr.1" = getelementptr i8, i8* %"stream_samplesRaw_arena_bytes.5", i32 %"stream_samplesRaw_byte_offset.5"
  %".32" = bitcast i8* %"stream_samplesRaw_flagged_byte_ptr.1" to i1*
  %"stream_samplesRaw_val.2" = load i1, i1* %".32"
  %".33" = insertvalue %"struct.Sample" %".31", i1 %"stream_samplesRaw_val.2", 2
  %".34" = call i1 @"filter_DropFlagged"(%"struct.Sample" %".33", %"struct.Sample"* %"fused_samplesKept_tail_slot")
  br i1 %".34", label %"fused_0_tail_kept_0", label %"fused_0_tail_row_done"
fused_0_tail_exit:
  %"fused_carry_final_vec" = load <8 x float>, <8 x float>* %"fused_0_keptTotal_vec"
  %"fused_carry_reduce" = call fast float @"llvm.vector.reduce.fadd.v8f32"(float              0x0, <8 x float> %"fused_carry_final_vec")
  %"fused_carry_final_tail" = load float, float* %"fused_0_keptTotal_tail"
  %"fused_carry_final" = fadd fast float %"fused_carry_reduce", %"fused_carry_final_tail"
  %"fused_0_kept" = load i32, i32* %"fused_0_write_idx"
  %"count_samplesScaled_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"count_samplesScaled_value_byte_ptr" = getelementptr i8, i8* %"count_samplesScaled_arena_bytes", i32 628
  %".59" = bitcast i8* %"count_samplesScaled_value_byte_ptr" to i32*
  store i32 %"fused_0_kept", i32* %".59"
  %"uniform_keptTotal_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"uniform_keptTotal_value_byte_ptr" = getelementptr i8, i8* %"uniform_keptTotal_arena_bytes", i32 620
  %".61" = bitcast i8* %"uniform_keptTotal_value_byte_ptr" to float*
  store float %"fused_carry_final", float* %".61"
  ret void
fused_0_tail_row_done:
  %"fused_tail_next" = add i32 %"fused_tail_idx", 1
  store i32 %"fused_tail_next", i32* %"fused_0_tail_idx"
  br label %"fused_0_tail_cond"
fused_0_tail_kept_0:
  %"fused_total_tail_acc" = alloca float
  store float 0.0, float* %"fused_total_tail_acc"
  %"fused_tail_write_idx" = load i32, i32* %"fused_0_write_idx"
  %"fused_samplesKept" = load %"struct.Sample", %"struct.Sample"* %"fused_samplesKept_tail_slot"
  %"route_clamp_lo_cmp.1" = icmp sgt i32 %"fused_tail_write_idx", 0
  %"route_clamp_lo.1" = select  i1 %"route_clamp_lo_cmp.1", i32 %"fused_tail_write_idx", i32 0
  %"route_clamp_hi_cmp.1" = icmp slt i32 %"route_clamp_lo.1", 19
  %"route_clamp.1" = select  i1 %"route_clamp_hi_cmp.1", i32 %"route_clamp_lo.1", i32 19
  %"route_samplesScaled_out_slot" = alloca %"struct.Sample"
  %"stream_samplesScaled_byte_index.6" = mul i32 %"fused_tail_write_idx", 4
  %"stream_samplesScaled_byte_offset.6" = add i32 360, %"stream_samplesScaled_byte_index.6"
  %"stream_samplesScaled_arena_bytes.6" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_id_byte_ptr.2" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.6", i32 %"stream_samplesScaled_byte_offset.6"
  %".37" = bitcast i8* %"stream_samplesScaled_id_byte_ptr.2" to i32*
  %"stream_samplesScaled_val" = load i32, i32* %".37"
  %".38" = insertvalue %"struct.Sample" undef, i32 %"stream_samplesScaled_val", 0
  %"stream_samplesScaled_byte_index.7" = mul i32 %"fused_tail_write_idx", 4
  %"stream_samplesScaled_byte_offset.7" = add i32 440, %"stream_samplesScaled_byte_index.7"
  %"stream_samplesScaled_arena_bytes.7" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_value_byte_ptr.2" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.7", i32 %"stream_samplesScaled_byte_offset.7"
  %".39" = bitcast i8* %"stream_samplesScaled_value_byte_ptr.2" to float*
  %"stream_samplesScaled_val.1" = load float, float* %".39"
  %".40" = insertvalue %"struct.Sample" %".38", float %"stream_samplesScaled_val.1", 1
  %"stream_samplesScaled_byte_index.8" = mul i32 %"fused_tail_write_idx", 1
  %"stream_samplesScaled_byte_offset.8" = add i32 520, %"stream_samplesScaled_byte_index.8"
  %"stream_samplesScaled_arena_bytes.8" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_flagged_byte_ptr.2" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.8", i32 %"stream_samplesScaled_byte_offset.8"
  %".41" = bitcast i8* %"stream_samplesScaled_flagged_byte_ptr.2" to i1*
  %"stream_samplesScaled_val.2" = load i1, i1* %".41"
  %".42" = insertvalue %"struct.Sample" %".40", i1 %"stream_samplesScaled_val.2", 2
  store %"struct.Sample" %".42", %"struct.Sample"* %"route_samplesScaled_out_slot"
  call void @"shader_Scale"(%"struct.Sample" %"fused_samplesKept", %"struct.Sample"* %"route_samplesScaled_out_slot", float* %"fused_total_tail_acc")
  %"route_samplesScaled_out_value" = load %"struct.Sample", %"struct.Sample"* %"route_samplesScaled_out_slot"
  %".45" = extractvalue %"struct.Sample" %"route_samplesScaled_out_value", 0
  %"stream_samplesScaled_byte_index.9" = mul i32 %"fused_tail_write_idx", 4
  %"stream_samplesScaled_byte_offset.9" = add i32 360, %"stream_samplesScaled_byte_index.9"
  %"stream_samplesScaled_arena_bytes.9" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_id_byte_ptr.3" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.9", i32 %"stream_samplesScaled_byte_offset.9"
  %".46" = bitcast i8* %"stream_samplesScaled_id_byte_ptr.3" to i32*
  store i32 %".45", i32* %".46"
  %".48" = extractvalue %"struct.Sample" %"route_samplesScaled_out_value", 1
  %"stream_samplesScaled_byte_index.10" = mul i32 %"fused_tail_write_idx", 4
  %"stream_samplesScaled_byte_offset.10" = add i32 440, %"stream_samplesScaled_byte_index.10"
  %"stream_samplesScaled_arena_bytes.10" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_value_byte_ptr.3" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.10", i32 %"stream_samplesScaled_byte_offset.10"
  %".49" = bitcast i8* %"stream_samplesScaled_value_byte_ptr.3" to float*
  store float %".48", float* %".49"
  %".51" = extractvalue %"struct.Sample" %"route_samplesScaled_out_value", 2
  %"stream_samplesScaled_byte_index.11" = mul i32 %"fused_tail_write_idx", 1
  %"stream_samplesScaled_byte_offset.11" = add i32 520, %"stream_samplesScaled_byte_index.11"
  %"stream_samplesScaled_arena_bytes.11" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_samplesScaled_flagged_byte_ptr.3" = getelementptr i8, i8* %"stream_samplesScaled_arena_bytes.11", i32 %"stream_samplesScaled_byte_offset.11"
  %".52" = bitcast i8* %"stream_samplesScaled_flagged_byte_ptr.3" to i1*
  store i1 %".51", i1* %".52"
  %"fused_tail_acc_val" = load float, float* %"fused_total_tail_acc"
  %"fused_tail_cur" = load float, float* %"fused_0_keptTotal_tail"
  %"fused_tail_next_acc" = fadd fast float %"fused_tail_cur", %"fused_tail_acc_val"
  store float %"fused_tail_next_acc", float* %"fused_0_keptTotal_tail"
  %"fused_tail_write_next" = add i32 %"fused_tail_write_idx", 1
  store i32 %"fused_tail_write_next", i32* %"fused_0_write_idx"
  br label %"fused_0_tail_row_done"
}

declare void @"llvm.masked.compressstore.v8i32"(<8 x i32> %".1", i32* %".2", <8 x i1> %".3")

declare void @"llvm.masked.compressstore.v8f32"(<8 x float> %".1", float* %".2", <8 x i1> %".3")

declare void @"llvm.masked.compressstore.v8i8"(<8 x i8> %".1", i8* %".2", <8 x i1> %".3")

declare i32 @"llvm.vector.reduce.add.v8i32"(<8 x i32> %".1")

declare float @"llvm.vector.reduce.fadd.v8f32"(float %".1", <8 x float> %".2")
