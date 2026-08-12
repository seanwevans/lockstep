; ModuleID = "lockstep"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%"struct.Event" = type {i32, float}
%"struct.Alert" = type {i32, float}
%"struct.Lockstep_Arena" = type {[4096 x i32], [4096 x float], [4096 x i32], [4096 x float], [4096 x i32], [4096 x float], [4096 x float]}
declare float @"pure_step"(float %"edge", float %"x")

declare float @"pure_mix"(float %"a", float %"b", float %"t")

declare float @"pure_clamp"(float %"x", float %"min_value", float %"max_value")

declare float @"pure_max"(float %"x", float %"y")

declare float @"pure_min"(float %"x", float %"y")

declare float @"pure_abs"(float %"x")

declare float @"pure_sign"(float %"x")

declare float @"pure_smoothstep"(float %"edge0", float %"edge1", float %"x")

define void @"shader_Normalize"(%"struct.Event" %"src", %"struct.Event"* %"dst")
{
entry:
  %"src.1" = alloca %"struct.Event"
  store %"struct.Event" %"src", %"struct.Event"* %"src.1"
  %"dst.1" = alloca %"struct.Event"*
  store %"struct.Event"* %"dst", %"struct.Event"** %"dst.1"
  %"src_val" = load %"struct.Event", %"struct.Event"* %"src.1"
  %"deviceId_field" = extractvalue %"struct.Event" %"src_val", 0
  %"dst_ptr" = load %"struct.Event"*, %"struct.Event"** %"dst.1"
  %"dst_ref" = load %"struct.Event", %"struct.Event"* %"dst_ptr"
  %"set_deviceId" = insertvalue %"struct.Event" %"dst_ref", i32 %"deviceId_field", 0
  %"dst_ptr.1" = load %"struct.Event"*, %"struct.Event"** %"dst.1"
  store %"struct.Event" %"set_deviceId", %"struct.Event"* %"dst_ptr.1"
  %"src_val.1" = load %"struct.Event", %"struct.Event"* %"src.1"
  %"value_field" = extractvalue %"struct.Event" %"src_val.1", 1
  %".7" = fmul float %"value_field", 0x3fb99999a0000000
  %"dst_ptr.2" = load %"struct.Event"*, %"struct.Event"** %"dst.1"
  %"dst_ref.1" = load %"struct.Event", %"struct.Event"* %"dst_ptr.2"
  %"set_value" = insertvalue %"struct.Event" %"dst_ref.1", float %".7", 1
  %"dst_ptr.3" = load %"struct.Event"*, %"struct.Event"** %"dst.1"
  store %"struct.Event" %"set_value", %"struct.Event"* %"dst_ptr.3"
  ret void
}

define void @"shader_Score"(%"struct.Event" %"src", %"struct.Alert"* %"dst", float* %"scoreSum")
{
entry:
  %"src.1" = alloca %"struct.Event"
  store %"struct.Event" %"src", %"struct.Event"* %"src.1"
  %"dst.1" = alloca %"struct.Alert"*
  store %"struct.Alert"* %"dst", %"struct.Alert"** %"dst.1"
  %"scoreSum.1" = alloca float*
  store float* %"scoreSum", float** %"scoreSum.1"
  %"score" = alloca float
  store float 0.0, float* %"score"
  %"src_val" = load %"struct.Event", %"struct.Event"* %"src.1"
  %"value_field" = extractvalue %"struct.Event" %"src_val", 1
  %".9" = fmul float %"value_field", 0x3ff99999a0000000
  store float %".9", float* %"score"
  %"scoreSum_val" = load float*, float** %"scoreSum.1"
  %"scoreSum_ref" = load float, float* %"scoreSum_val"
  %"score_val" = load float, float* %"score"
  %".11" = fadd float %"scoreSum_ref", %"score_val"
  %"scoreSum_ptr" = load float*, float** %"scoreSum.1"
  store float %".11", float* %"scoreSum_ptr"
  %"src_val.1" = load %"struct.Event", %"struct.Event"* %"src.1"
  %"deviceId_field" = extractvalue %"struct.Event" %"src_val.1", 0
  %"dst_ptr" = load %"struct.Alert"*, %"struct.Alert"** %"dst.1"
  %"dst_ref" = load %"struct.Alert", %"struct.Alert"* %"dst_ptr"
  %"set_deviceId" = insertvalue %"struct.Alert" %"dst_ref", i32 %"deviceId_field", 0
  %"dst_ptr.1" = load %"struct.Alert"*, %"struct.Alert"** %"dst.1"
  store %"struct.Alert" %"set_deviceId", %"struct.Alert"* %"dst_ptr.1"
  %"score_val.1" = load float, float* %"score"
  %"dst_ptr.2" = load %"struct.Alert"*, %"struct.Alert"** %"dst.1"
  %"dst_ref.1" = load %"struct.Alert", %"struct.Alert"* %"dst_ptr.2"
  %"set_score" = insertvalue %"struct.Alert" %"dst_ref.1", float %"score_val.1", 1
  %"dst_ptr.3" = load %"struct.Alert"*, %"struct.Alert"** %"dst.1"
  store %"struct.Alert" %"set_score", %"struct.Alert"* %"dst_ptr.3"
  ret void
}

define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* noalias nocapture %"arena")
{
entry:
  %"fused_0_idx" = alloca i32
  store i32 0, i32* %"fused_0_idx"
  %"fused_0_totalScore_vec" = alloca <8 x float>
  store <8 x float> <float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0, float              0x0>, <8 x float>* %"fused_0_totalScore_vec"
  %"fused_0_totalScore_tail" = alloca float
  store float              0x0, float* %"fused_0_totalScore_tail"
  br label %"fused_0_cond"
fused_0_cond:
  %"fused_idx" = load i32, i32* %"fused_0_idx"
  %"fused_vector_active" = icmp slt i32 %"fused_idx", 4096
  br i1 %"fused_vector_active", label %"fused_0_body", label %"fused_0_exit"
fused_0_body:
  %"stream_eventsRaw_byte_index" = mul i32 %"fused_idx", 4
  %"stream_eventsRaw_byte_offset" = add i32 0, %"stream_eventsRaw_byte_index"
  %"stream_eventsRaw_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_eventsRaw_deviceId_byte_ptr" = getelementptr i8, i8* %"stream_eventsRaw_arena_bytes", i32 %"stream_eventsRaw_byte_offset"
  %".8" = bitcast i8* %"stream_eventsRaw_deviceId_byte_ptr" to i32*
  %"fused_eventsRaw_deviceId_vector_ptr" = bitcast i32* %".8" to <8 x i32>*
  %"fused_eventsRaw_deviceId_vector" = load <8 x i32>, <8 x i32>* %"fused_eventsRaw_deviceId_vector_ptr", align 1
  %"stream_eventsRaw_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_eventsRaw_byte_offset.1" = add i32 16384, %"stream_eventsRaw_byte_index.1"
  %"stream_eventsRaw_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_eventsRaw_value_byte_ptr" = getelementptr i8, i8* %"stream_eventsRaw_arena_bytes.1", i32 %"stream_eventsRaw_byte_offset.1"
  %".9" = bitcast i8* %"stream_eventsRaw_value_byte_ptr" to float*
  %"fused_eventsRaw_value_vector_ptr" = bitcast float* %".9" to <8 x float>*
  %"fused_eventsRaw_value_vector" = load <8 x float>, <8 x float>* %"fused_eventsRaw_value_vector_ptr", align 1
  %"stream_eventsEnriched_byte_index" = mul i32 %"fused_idx", 4
  %"stream_eventsEnriched_byte_offset" = add i32 32768, %"stream_eventsEnriched_byte_index"
  %"stream_eventsEnriched_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_eventsEnriched_deviceId_byte_ptr" = getelementptr i8, i8* %"stream_eventsEnriched_arena_bytes", i32 %"stream_eventsEnriched_byte_offset"
  %".10" = bitcast i8* %"stream_eventsEnriched_deviceId_byte_ptr" to i32*
  %"fused_eventsEnriched_deviceId_vector_ptr" = bitcast i32* %".10" to <8 x i32>*
  %"fused_eventsEnriched_deviceId_vector" = load <8 x i32>, <8 x i32>* %"fused_eventsEnriched_deviceId_vector_ptr", align 1
  %"stream_eventsEnriched_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_eventsEnriched_byte_offset.1" = add i32 49152, %"stream_eventsEnriched_byte_index.1"
  %"stream_eventsEnriched_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_eventsEnriched_value_byte_ptr" = getelementptr i8, i8* %"stream_eventsEnriched_arena_bytes.1", i32 %"stream_eventsEnriched_byte_offset.1"
  %".11" = bitcast i8* %"stream_eventsEnriched_value_byte_ptr" to float*
  %"fused_eventsEnriched_value_vector_ptr" = bitcast float* %".11" to <8 x float>*
  %"fused_eventsEnriched_value_vector" = load <8 x float>, <8 x float>* %"fused_eventsEnriched_value_vector_ptr", align 1
  %".12" = insertelement <8 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef>, float 0x3fb99999a0000000, i32 0
  %"fused_splat" = shufflevector <8 x float> %".12", <8 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %"fused_math" = fmul <8 x float> %"fused_eventsRaw_value_vector", %"fused_splat"
  %"stream_alertsScored_byte_index" = mul i32 %"fused_idx", 4
  %"stream_alertsScored_byte_offset" = add i32 65536, %"stream_alertsScored_byte_index"
  %"stream_alertsScored_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_alertsScored_deviceId_byte_ptr" = getelementptr i8, i8* %"stream_alertsScored_arena_bytes", i32 %"stream_alertsScored_byte_offset"
  %".13" = bitcast i8* %"stream_alertsScored_deviceId_byte_ptr" to i32*
  %"fused_alertsScored_deviceId_vector_ptr" = bitcast i32* %".13" to <8 x i32>*
  %"fused_alertsScored_deviceId_vector" = load <8 x i32>, <8 x i32>* %"fused_alertsScored_deviceId_vector_ptr", align 1
  %"stream_alertsScored_byte_index.1" = mul i32 %"fused_idx", 4
  %"stream_alertsScored_byte_offset.1" = add i32 81920, %"stream_alertsScored_byte_index.1"
  %"stream_alertsScored_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_alertsScored_score_byte_ptr" = getelementptr i8, i8* %"stream_alertsScored_arena_bytes.1", i32 %"stream_alertsScored_byte_offset.1"
  %".14" = bitcast i8* %"stream_alertsScored_score_byte_ptr" to float*
  %"fused_alertsScored_score_vector_ptr" = bitcast float* %".14" to <8 x float>*
  %"fused_alertsScored_score_vector" = load <8 x float>, <8 x float>* %"fused_alertsScored_score_vector_ptr", align 1
  %".15" = insertelement <8 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef>, float 0x3ff99999a0000000, i32 0
  %"fused_splat.1" = shufflevector <8 x float> %".15", <8 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %"fused_math.1" = fmul <8 x float> %"fused_math", %"fused_splat.1"
  %"fused_math.2" = fadd <8 x float> <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, %"fused_math.1"
  %"stream_alertsScored_byte_index.2" = mul i32 %"fused_idx", 4
  %"stream_alertsScored_byte_offset.2" = add i32 65536, %"stream_alertsScored_byte_index.2"
  %"stream_alertsScored_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_alertsScored_deviceId_byte_ptr.1" = getelementptr i8, i8* %"stream_alertsScored_arena_bytes.2", i32 %"stream_alertsScored_byte_offset.2"
  %".16" = bitcast i8* %"stream_alertsScored_deviceId_byte_ptr.1" to i32*
  %"fused_alertsScored_deviceId_vector_ptr.1" = bitcast i32* %".16" to <8 x i32>*
  store <8 x i32> %"fused_eventsRaw_deviceId_vector", <8 x i32>* %"fused_alertsScored_deviceId_vector_ptr.1", align 1
  %"stream_alertsScored_byte_index.3" = mul i32 %"fused_idx", 4
  %"stream_alertsScored_byte_offset.3" = add i32 81920, %"stream_alertsScored_byte_index.3"
  %"stream_alertsScored_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_alertsScored_score_byte_ptr.1" = getelementptr i8, i8* %"stream_alertsScored_arena_bytes.3", i32 %"stream_alertsScored_byte_offset.3"
  %".18" = bitcast i8* %"stream_alertsScored_score_byte_ptr.1" to float*
  %"fused_alertsScored_score_vector_ptr.1" = bitcast float* %".18" to <8 x float>*
  store <8 x float> %"fused_math.1", <8 x float>* %"fused_alertsScored_score_vector_ptr.1", align 1
  %"fused_carry_cur" = load <8 x float>, <8 x float>* %"fused_0_totalScore_vec"
  %"fused_carry_next" = fadd fast <8 x float> %"fused_carry_cur", %"fused_math.2"
  store <8 x float> %"fused_carry_next", <8 x float>* %"fused_0_totalScore_vec"
  %"fused_idx_next" = add i32 %"fused_idx", 8
  store i32 %"fused_idx_next", i32* %"fused_0_idx"
  br label %"fused_0_cond"
fused_0_exit:
  %"fused_carry_final_vec" = load <8 x float>, <8 x float>* %"fused_0_totalScore_vec"
  %"fused_carry_reduce" = call fast float @"llvm.vector.reduce.fadd.v8f32"(float              0x0, <8 x float> %"fused_carry_final_vec")
  %"fused_carry_final_tail" = load float, float* %"fused_0_totalScore_tail"
  %"fused_carry_final" = fadd fast float %"fused_carry_reduce", %"fused_carry_final_tail"
  ret void
}

declare float @"llvm.vector.reduce.fadd.v8f32"(float %".1", <8 x float> %".2")
