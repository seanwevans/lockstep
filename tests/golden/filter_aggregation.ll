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
  %"DropInvalid_idx" = alloca i32
  store i32 0, i32* %"DropInvalid_idx"
  %"DropInvalid_write_idx" = alloca i32
  store i32 0, i32* %"DropInvalid_write_idx"
  br label %"route_DropInvalid_cond"
route_DropInvalid_cond:
  %"idx" = load i32, i32* %"DropInvalid_idx"
  %"route_active" = icmp slt i32 %"idx", 4096
  br i1 %"route_active", label %"route_DropInvalid_body", label %"route_DropInvalid_exit"
route_DropInvalid_body:
  %"filter_write_idx" = load i32, i32* %"DropInvalid_write_idx"
  %".7" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx", i32 0
  %"route_i32_splat" = shufflevector <4 x i32> %".7", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".8" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.1" = shufflevector <4 x i32> %".8", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp" = icmp sgt <4 x i32> %"route_i32_splat", %"route_i32_splat.1"
  %"route_vec_max" = select  <4 x i1> %"route_vec_max_cmp", <4 x i32> %"route_i32_splat", <4 x i32> %"route_i32_splat.1"
  %"route_i32_lane0" = extractelement <4 x i32> %"route_vec_max", i32 0
  %".9" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0", i32 0
  %"route_i32_splat.2" = shufflevector <4 x i32> %".9", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".10" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 4095, i32 0
  %"route_i32_splat.3" = shufflevector <4 x i32> %".10", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp" = icmp slt <4 x i32> %"route_i32_splat.2", %"route_i32_splat.3"
  %"route_vec_min" = select  <4 x i1> %"route_vec_min_cmp", <4 x i32> %"route_i32_splat.2", <4 x i32> %"route_i32_splat.3"
  %"route_i32_lane0.1" = extractelement <4 x i32> %"route_vec_min", i32 0
  %"stream_readingsRaw_byte_index" = mul i32 %"route_i32_lane0.1", 4
  %"stream_readingsRaw_byte_offset" = add i32 0, %"stream_readingsRaw_byte_index"
  %"stream_readingsRaw_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsRaw_sensorId_byte_ptr" = getelementptr i8, i8* %"stream_readingsRaw_arena_bytes", i32 %"stream_readingsRaw_byte_offset"
  %".11" = bitcast i8* %"stream_readingsRaw_sensorId_byte_ptr" to i32*
  %"stream_readingsRaw_val" = load i32, i32* %".11"
  %".12" = insertvalue %"struct.Reading" undef, i32 %"stream_readingsRaw_val", 0
  %"stream_readingsRaw_byte_index.1" = mul i32 %"route_i32_lane0.1", 4
  %"stream_readingsRaw_byte_offset.1" = add i32 16384, %"stream_readingsRaw_byte_index.1"
  %"stream_readingsRaw_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsRaw_value_byte_ptr" = getelementptr i8, i8* %"stream_readingsRaw_arena_bytes.1", i32 %"stream_readingsRaw_byte_offset.1"
  %".13" = bitcast i8* %"stream_readingsRaw_value_byte_ptr" to float*
  %"stream_readingsRaw_val.1" = load float, float* %".13"
  %".14" = insertvalue %"struct.Reading" %".12", float %"stream_readingsRaw_val.1", 1
  %"stream_readingsRaw_byte_index.2" = mul i32 %"route_i32_lane0.1", 1
  %"stream_readingsRaw_byte_offset.2" = add i32 32768, %"stream_readingsRaw_byte_index.2"
  %"stream_readingsRaw_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsRaw_valid_byte_ptr" = getelementptr i8, i8* %"stream_readingsRaw_arena_bytes.2", i32 %"stream_readingsRaw_byte_offset.2"
  %".15" = bitcast i8* %"stream_readingsRaw_valid_byte_ptr" to i1*
  %"stream_readingsRaw_val.2" = load i1, i1* %".15"
  %".16" = insertvalue %"struct.Reading" %".14", i1 %"stream_readingsRaw_val.2", 2
  %"route_readingsValid_out_slot" = alloca %"struct.Reading"
  %"stream_readingsValid_byte_index" = mul i32 %"filter_write_idx", 4
  %"stream_readingsValid_byte_offset" = add i32 36864, %"stream_readingsValid_byte_index"
  %"stream_readingsValid_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_sensorId_byte_ptr" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes", i32 %"stream_readingsValid_byte_offset"
  %".17" = bitcast i8* %"stream_readingsValid_sensorId_byte_ptr" to i32*
  %"stream_readingsValid_val" = load i32, i32* %".17"
  %".18" = insertvalue %"struct.Reading" undef, i32 %"stream_readingsValid_val", 0
  %"stream_readingsValid_byte_index.1" = mul i32 %"filter_write_idx", 4
  %"stream_readingsValid_byte_offset.1" = add i32 53248, %"stream_readingsValid_byte_index.1"
  %"stream_readingsValid_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_value_byte_ptr" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.1", i32 %"stream_readingsValid_byte_offset.1"
  %".19" = bitcast i8* %"stream_readingsValid_value_byte_ptr" to float*
  %"stream_readingsValid_val.1" = load float, float* %".19"
  %".20" = insertvalue %"struct.Reading" %".18", float %"stream_readingsValid_val.1", 1
  %"stream_readingsValid_byte_index.2" = mul i32 %"filter_write_idx", 1
  %"stream_readingsValid_byte_offset.2" = add i32 69632, %"stream_readingsValid_byte_index.2"
  %"stream_readingsValid_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_valid_byte_ptr" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.2", i32 %"stream_readingsValid_byte_offset.2"
  %".21" = bitcast i8* %"stream_readingsValid_valid_byte_ptr" to i1*
  %"stream_readingsValid_val.2" = load i1, i1* %".21"
  %".22" = insertvalue %"struct.Reading" %".20", i1 %"stream_readingsValid_val.2", 2
  store %"struct.Reading" %".22", %"struct.Reading"* %"route_readingsValid_out_slot"
  %".24" = call i1 @"filter_DropInvalid"(%"struct.Reading" %".16", %"struct.Reading"* %"route_readingsValid_out_slot")
  br i1 %".24", label %"filter_DropInvalid_store", label %"filter_DropInvalid_skip"
route_DropInvalid_exit:
  %"Aggregate_idx" = alloca i32
  store i32 0, i32* %"Aggregate_idx"
  br label %"route_Aggregate_cond"
filter_DropInvalid_store:
  %"route_readingsValid_out_value" = load %"struct.Reading", %"struct.Reading"* %"route_readingsValid_out_slot"
  %".26" = extractvalue %"struct.Reading" %"route_readingsValid_out_value", 0
  %"stream_readingsValid_byte_index.3" = mul i32 %"filter_write_idx", 4
  %"stream_readingsValid_byte_offset.3" = add i32 36864, %"stream_readingsValid_byte_index.3"
  %"stream_readingsValid_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_sensorId_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.3", i32 %"stream_readingsValid_byte_offset.3"
  %".27" = bitcast i8* %"stream_readingsValid_sensorId_byte_ptr.1" to i32*
  store i32 %".26", i32* %".27"
  %".29" = extractvalue %"struct.Reading" %"route_readingsValid_out_value", 1
  %"stream_readingsValid_byte_index.4" = mul i32 %"filter_write_idx", 4
  %"stream_readingsValid_byte_offset.4" = add i32 53248, %"stream_readingsValid_byte_index.4"
  %"stream_readingsValid_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_value_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.4", i32 %"stream_readingsValid_byte_offset.4"
  %".30" = bitcast i8* %"stream_readingsValid_value_byte_ptr.1" to float*
  store float %".29", float* %".30"
  %".32" = extractvalue %"struct.Reading" %"route_readingsValid_out_value", 2
  %"stream_readingsValid_byte_index.5" = mul i32 %"filter_write_idx", 1
  %"stream_readingsValid_byte_offset.5" = add i32 69632, %"stream_readingsValid_byte_index.5"
  %"stream_readingsValid_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_valid_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.5", i32 %"stream_readingsValid_byte_offset.5"
  %".33" = bitcast i8* %"stream_readingsValid_valid_byte_ptr.1" to i1*
  store i1 %".32", i1* %".33"
  br label %"filter_DropInvalid_after"
filter_DropInvalid_skip:
  br label %"filter_DropInvalid_after"
filter_DropInvalid_after:
  %"filter_write_next" = add i32 %"filter_write_idx", 1
  %"filter_write_select" = select  i1 %".24", i32 %"filter_write_next", i32 %"filter_write_idx"
  store i32 %"filter_write_select", i32* %"DropInvalid_write_idx"
  %"idx_next" = add i32 %"idx", 1
  store i32 %"idx_next", i32* %"DropInvalid_idx"
  br label %"route_DropInvalid_cond"
route_Aggregate_cond:
  %"idx.1" = load i32, i32* %"Aggregate_idx"
  %"route_active.1" = icmp slt i32 %"idx.1", 4096
  br i1 %"route_active.1", label %"route_Aggregate_body", label %"route_Aggregate_exit"
route_Aggregate_body:
  %".43" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx.1", i32 0
  %"route_i32_splat.4" = shufflevector <4 x i32> %".43", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".44" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.5" = shufflevector <4 x i32> %".44", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp.1" = icmp sgt <4 x i32> %"route_i32_splat.4", %"route_i32_splat.5"
  %"route_vec_max.1" = select  <4 x i1> %"route_vec_max_cmp.1", <4 x i32> %"route_i32_splat.4", <4 x i32> %"route_i32_splat.5"
  %"route_i32_lane0.2" = extractelement <4 x i32> %"route_vec_max.1", i32 0
  %".45" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0.2", i32 0
  %"route_i32_splat.6" = shufflevector <4 x i32> %".45", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".46" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 4095, i32 0
  %"route_i32_splat.7" = shufflevector <4 x i32> %".46", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp.1" = icmp slt <4 x i32> %"route_i32_splat.6", %"route_i32_splat.7"
  %"route_vec_min.1" = select  <4 x i1> %"route_vec_min_cmp.1", <4 x i32> %"route_i32_splat.6", <4 x i32> %"route_i32_splat.7"
  %"route_i32_lane0.3" = extractelement <4 x i32> %"route_vec_min.1", i32 0
  %"stream_readingsValid_byte_index.6" = mul i32 %"route_i32_lane0.3", 4
  %"stream_readingsValid_byte_offset.6" = add i32 36864, %"stream_readingsValid_byte_index.6"
  %"stream_readingsValid_arena_bytes.6" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_sensorId_byte_ptr.2" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.6", i32 %"stream_readingsValid_byte_offset.6"
  %".47" = bitcast i8* %"stream_readingsValid_sensorId_byte_ptr.2" to i32*
  %"stream_readingsValid_val.3" = load i32, i32* %".47"
  %".48" = insertvalue %"struct.Reading" undef, i32 %"stream_readingsValid_val.3", 0
  %"stream_readingsValid_byte_index.7" = mul i32 %"route_i32_lane0.3", 4
  %"stream_readingsValid_byte_offset.7" = add i32 53248, %"stream_readingsValid_byte_index.7"
  %"stream_readingsValid_arena_bytes.7" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_value_byte_ptr.2" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.7", i32 %"stream_readingsValid_byte_offset.7"
  %".49" = bitcast i8* %"stream_readingsValid_value_byte_ptr.2" to float*
  %"stream_readingsValid_val.4" = load float, float* %".49"
  %".50" = insertvalue %"struct.Reading" %".48", float %"stream_readingsValid_val.4", 1
  %"stream_readingsValid_byte_index.8" = mul i32 %"route_i32_lane0.3", 1
  %"stream_readingsValid_byte_offset.8" = add i32 69632, %"stream_readingsValid_byte_index.8"
  %"stream_readingsValid_arena_bytes.8" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsValid_valid_byte_ptr.2" = getelementptr i8, i8* %"stream_readingsValid_arena_bytes.8", i32 %"stream_readingsValid_byte_offset.8"
  %".51" = bitcast i8* %"stream_readingsValid_valid_byte_ptr.2" to i1*
  %"stream_readingsValid_val.5" = load i1, i1* %".51"
  %".52" = insertvalue %"struct.Reading" %".50", i1 %"stream_readingsValid_val.5", 2
  %".53" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx.1", i32 0
  %"route_i32_splat.8" = shufflevector <4 x i32> %".53", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".54" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.9" = shufflevector <4 x i32> %".54", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp.2" = icmp sgt <4 x i32> %"route_i32_splat.8", %"route_i32_splat.9"
  %"route_vec_max.2" = select  <4 x i1> %"route_vec_max_cmp.2", <4 x i32> %"route_i32_splat.8", <4 x i32> %"route_i32_splat.9"
  %"route_i32_lane0.4" = extractelement <4 x i32> %"route_vec_max.2", i32 0
  %".55" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0.4", i32 0
  %"route_i32_splat.10" = shufflevector <4 x i32> %".55", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".56" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 4095, i32 0
  %"route_i32_splat.11" = shufflevector <4 x i32> %".56", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp.2" = icmp slt <4 x i32> %"route_i32_splat.10", %"route_i32_splat.11"
  %"route_vec_min.2" = select  <4 x i1> %"route_vec_min_cmp.2", <4 x i32> %"route_i32_splat.10", <4 x i32> %"route_i32_splat.11"
  %"route_i32_lane0.5" = extractelement <4 x i32> %"route_vec_min.2", i32 0
  %"route_readingsAggregated_out_slot" = alloca %"struct.Reading"
  %"stream_readingsAggregated_byte_index" = mul i32 %"idx.1", 4
  %"stream_readingsAggregated_byte_offset" = add i32 73728, %"stream_readingsAggregated_byte_index"
  %"stream_readingsAggregated_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_sensorId_byte_ptr" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes", i32 %"stream_readingsAggregated_byte_offset"
  %".57" = bitcast i8* %"stream_readingsAggregated_sensorId_byte_ptr" to i32*
  %"stream_readingsAggregated_val" = load i32, i32* %".57"
  %".58" = insertvalue %"struct.Reading" undef, i32 %"stream_readingsAggregated_val", 0
  %"stream_readingsAggregated_byte_index.1" = mul i32 %"idx.1", 4
  %"stream_readingsAggregated_byte_offset.1" = add i32 90112, %"stream_readingsAggregated_byte_index.1"
  %"stream_readingsAggregated_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_value_byte_ptr" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.1", i32 %"stream_readingsAggregated_byte_offset.1"
  %".59" = bitcast i8* %"stream_readingsAggregated_value_byte_ptr" to float*
  %"stream_readingsAggregated_val.1" = load float, float* %".59"
  %".60" = insertvalue %"struct.Reading" %".58", float %"stream_readingsAggregated_val.1", 1
  %"stream_readingsAggregated_byte_index.2" = mul i32 %"idx.1", 1
  %"stream_readingsAggregated_byte_offset.2" = add i32 106496, %"stream_readingsAggregated_byte_index.2"
  %"stream_readingsAggregated_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_valid_byte_ptr" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.2", i32 %"stream_readingsAggregated_byte_offset.2"
  %".61" = bitcast i8* %"stream_readingsAggregated_valid_byte_ptr" to i1*
  %"stream_readingsAggregated_val.2" = load i1, i1* %".61"
  %".62" = insertvalue %"struct.Reading" %".60", i1 %"stream_readingsAggregated_val.2", 2
  store %"struct.Reading" %".62", %"struct.Reading"* %"route_readingsAggregated_out_slot"
  %"accum_total_byte_index" = mul i32 %"idx.1", 4
  %"accum_total_byte_offset" = add i32 110592, %"accum_total_byte_index"
  %"accum_total_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"accum_total_value_byte_ptr" = getelementptr i8, i8* %"accum_total_arena_bytes", i32 %"accum_total_byte_offset"
  %".64" = bitcast i8* %"accum_total_value_byte_ptr" to float*
  call void @"shader_Aggregate"(%"struct.Reading" %".52", %"struct.Reading"* %"route_readingsAggregated_out_slot", float* %".64")
  %"route_readingsAggregated_out_value" = load %"struct.Reading", %"struct.Reading"* %"route_readingsAggregated_out_slot"
  %".66" = extractvalue %"struct.Reading" %"route_readingsAggregated_out_value", 0
  %"stream_readingsAggregated_byte_index.3" = mul i32 %"idx.1", 4
  %"stream_readingsAggregated_byte_offset.3" = add i32 73728, %"stream_readingsAggregated_byte_index.3"
  %"stream_readingsAggregated_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_sensorId_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.3", i32 %"stream_readingsAggregated_byte_offset.3"
  %".67" = bitcast i8* %"stream_readingsAggregated_sensorId_byte_ptr.1" to i32*
  store i32 %".66", i32* %".67"
  %".69" = extractvalue %"struct.Reading" %"route_readingsAggregated_out_value", 1
  %"stream_readingsAggregated_byte_index.4" = mul i32 %"idx.1", 4
  %"stream_readingsAggregated_byte_offset.4" = add i32 90112, %"stream_readingsAggregated_byte_index.4"
  %"stream_readingsAggregated_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_value_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.4", i32 %"stream_readingsAggregated_byte_offset.4"
  %".70" = bitcast i8* %"stream_readingsAggregated_value_byte_ptr.1" to float*
  store float %".69", float* %".70"
  %".72" = extractvalue %"struct.Reading" %"route_readingsAggregated_out_value", 2
  %"stream_readingsAggregated_byte_index.5" = mul i32 %"idx.1", 1
  %"stream_readingsAggregated_byte_offset.5" = add i32 106496, %"stream_readingsAggregated_byte_index.5"
  %"stream_readingsAggregated_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_readingsAggregated_valid_byte_ptr.1" = getelementptr i8, i8* %"stream_readingsAggregated_arena_bytes.5", i32 %"stream_readingsAggregated_byte_offset.5"
  %".73" = bitcast i8* %"stream_readingsAggregated_valid_byte_ptr.1" to i1*
  store i1 %".72", i1* %".73"
  %"idx_next.1" = add i32 %"idx.1", 1
  store i32 %"idx_next.1", i32* %"Aggregate_idx"
  br label %"route_Aggregate_cond"
route_Aggregate_exit:
  ret void
}
