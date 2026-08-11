; ModuleID = "lockstep"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%"struct.Pixel" = type {float, float, float}
%"struct.Lockstep_Arena" = type {[1024 x float], [1024 x float], [1024 x float], [1024 x float], [1024 x float], [1024 x float], float}
declare float @"pure_step"(float %"edge", float %"x")

declare float @"pure_mix"(float %"a", float %"b", float %"t")

declare float @"pure_clamp"(float %"x", float %"min_value", float %"max_value")

declare float @"pure_max"(float %"x", float %"y")

declare float @"pure_min"(float %"x", float %"y")

declare float @"pure_abs"(float %"x")

declare float @"pure_sign"(float %"x")

declare float @"pure_smoothstep"(float %"edge0", float %"edge1", float %"x")

define void @"shader_Brighten"(%"struct.Pixel" %"src", %"struct.Pixel"* %"dst", float %"gain")
{
entry:
  %"src.1" = alloca %"struct.Pixel"
  store %"struct.Pixel" %"src", %"struct.Pixel"* %"src.1"
  %"dst.1" = alloca %"struct.Pixel"*
  store %"struct.Pixel"* %"dst", %"struct.Pixel"** %"dst.1"
  %"gain.1" = alloca float
  store float %"gain", float* %"gain.1"
  %"src_val" = load %"struct.Pixel", %"struct.Pixel"* %"src.1"
  %"r_field" = extractvalue %"struct.Pixel" %"src_val", 0
  %"gain_val" = load float, float* %"gain.1"
  %".8" = fmul float %"r_field", %"gain_val"
  %"dst_ptr" = load %"struct.Pixel"*, %"struct.Pixel"** %"dst.1"
  %"dst_ref" = load %"struct.Pixel", %"struct.Pixel"* %"dst_ptr"
  %"set_r" = insertvalue %"struct.Pixel" %"dst_ref", float %".8", 0
  %"dst_ptr.1" = load %"struct.Pixel"*, %"struct.Pixel"** %"dst.1"
  store %"struct.Pixel" %"set_r", %"struct.Pixel"* %"dst_ptr.1"
  %"src_val.1" = load %"struct.Pixel", %"struct.Pixel"* %"src.1"
  %"g_field" = extractvalue %"struct.Pixel" %"src_val.1", 1
  %"gain_val.1" = load float, float* %"gain.1"
  %".10" = fmul float %"g_field", %"gain_val.1"
  %"dst_ptr.2" = load %"struct.Pixel"*, %"struct.Pixel"** %"dst.1"
  %"dst_ref.1" = load %"struct.Pixel", %"struct.Pixel"* %"dst_ptr.2"
  %"set_g" = insertvalue %"struct.Pixel" %"dst_ref.1", float %".10", 1
  %"dst_ptr.3" = load %"struct.Pixel"*, %"struct.Pixel"** %"dst.1"
  store %"struct.Pixel" %"set_g", %"struct.Pixel"* %"dst_ptr.3"
  %"src_val.2" = load %"struct.Pixel", %"struct.Pixel"* %"src.1"
  %"b_field" = extractvalue %"struct.Pixel" %"src_val.2", 2
  %"gain_val.2" = load float, float* %"gain.1"
  %".12" = fmul float %"b_field", %"gain_val.2"
  %"dst_ptr.4" = load %"struct.Pixel"*, %"struct.Pixel"** %"dst.1"
  %"dst_ref.2" = load %"struct.Pixel", %"struct.Pixel"* %"dst_ptr.4"
  %"set_b" = insertvalue %"struct.Pixel" %"dst_ref.2", float %".12", 2
  %"dst_ptr.5" = load %"struct.Pixel"*, %"struct.Pixel"** %"dst.1"
  store %"struct.Pixel" %"set_b", %"struct.Pixel"* %"dst_ptr.5"
  ret void
}

define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* %"arena")
{
entry:
  %"Brighten_idx" = alloca i32
  store i32 0, i32* %"Brighten_idx"
  br label %"route_Brighten_cond"
route_Brighten_cond:
  %"idx" = load i32, i32* %"Brighten_idx"
  %"route_active" = icmp slt i32 %"idx", 1024
  br i1 %"route_active", label %"route_Brighten_body", label %"route_Brighten_exit"
route_Brighten_body:
  %".6" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx", i32 0
  %"route_i32_splat" = shufflevector <4 x i32> %".6", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".7" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.1" = shufflevector <4 x i32> %".7", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp" = icmp sgt <4 x i32> %"route_i32_splat", %"route_i32_splat.1"
  %"route_vec_max" = select  <4 x i1> %"route_vec_max_cmp", <4 x i32> %"route_i32_splat", <4 x i32> %"route_i32_splat.1"
  %"route_i32_lane0" = extractelement <4 x i32> %"route_vec_max", i32 0
  %".8" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0", i32 0
  %"route_i32_splat.2" = shufflevector <4 x i32> %".8", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".9" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 1023, i32 0
  %"route_i32_splat.3" = shufflevector <4 x i32> %".9", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp" = icmp slt <4 x i32> %"route_i32_splat.2", %"route_i32_splat.3"
  %"route_vec_min" = select  <4 x i1> %"route_vec_min_cmp", <4 x i32> %"route_i32_splat.2", <4 x i32> %"route_i32_splat.3"
  %"route_i32_lane0.1" = extractelement <4 x i32> %"route_vec_min", i32 0
  %"stream_pixelsIn_byte_index" = mul i32 %"route_i32_lane0.1", 4
  %"stream_pixelsIn_byte_offset" = add i32 0, %"stream_pixelsIn_byte_index"
  %"stream_pixelsIn_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsIn_r_byte_ptr" = getelementptr i8, i8* %"stream_pixelsIn_arena_bytes", i32 %"stream_pixelsIn_byte_offset"
  %".10" = bitcast i8* %"stream_pixelsIn_r_byte_ptr" to float*
  %"stream_pixelsIn_val" = load float, float* %".10"
  %".11" = insertvalue %"struct.Pixel" undef, float %"stream_pixelsIn_val", 0
  %"stream_pixelsIn_byte_index.1" = mul i32 %"route_i32_lane0.1", 4
  %"stream_pixelsIn_byte_offset.1" = add i32 4096, %"stream_pixelsIn_byte_index.1"
  %"stream_pixelsIn_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsIn_g_byte_ptr" = getelementptr i8, i8* %"stream_pixelsIn_arena_bytes.1", i32 %"stream_pixelsIn_byte_offset.1"
  %".12" = bitcast i8* %"stream_pixelsIn_g_byte_ptr" to float*
  %"stream_pixelsIn_val.1" = load float, float* %".12"
  %".13" = insertvalue %"struct.Pixel" %".11", float %"stream_pixelsIn_val.1", 1
  %"stream_pixelsIn_byte_index.2" = mul i32 %"route_i32_lane0.1", 4
  %"stream_pixelsIn_byte_offset.2" = add i32 8192, %"stream_pixelsIn_byte_index.2"
  %"stream_pixelsIn_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsIn_b_byte_ptr" = getelementptr i8, i8* %"stream_pixelsIn_arena_bytes.2", i32 %"stream_pixelsIn_byte_offset.2"
  %".14" = bitcast i8* %"stream_pixelsIn_b_byte_ptr" to float*
  %"stream_pixelsIn_val.2" = load float, float* %".14"
  %".15" = insertvalue %"struct.Pixel" %".13", float %"stream_pixelsIn_val.2", 2
  %".16" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx", i32 0
  %"route_i32_splat.4" = shufflevector <4 x i32> %".16", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".17" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.5" = shufflevector <4 x i32> %".17", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp.1" = icmp sgt <4 x i32> %"route_i32_splat.4", %"route_i32_splat.5"
  %"route_vec_max.1" = select  <4 x i1> %"route_vec_max_cmp.1", <4 x i32> %"route_i32_splat.4", <4 x i32> %"route_i32_splat.5"
  %"route_i32_lane0.2" = extractelement <4 x i32> %"route_vec_max.1", i32 0
  %".18" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0.2", i32 0
  %"route_i32_splat.6" = shufflevector <4 x i32> %".18", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".19" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 1023, i32 0
  %"route_i32_splat.7" = shufflevector <4 x i32> %".19", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp.1" = icmp slt <4 x i32> %"route_i32_splat.6", %"route_i32_splat.7"
  %"route_vec_min.1" = select  <4 x i1> %"route_vec_min_cmp.1", <4 x i32> %"route_i32_splat.6", <4 x i32> %"route_i32_splat.7"
  %"route_i32_lane0.3" = extractelement <4 x i32> %"route_vec_min.1", i32 0
  %"route_pixelsOut_out_slot" = alloca %"struct.Pixel"
  %"stream_pixelsOut_byte_index" = mul i32 %"idx", 4
  %"stream_pixelsOut_byte_offset" = add i32 12288, %"stream_pixelsOut_byte_index"
  %"stream_pixelsOut_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsOut_r_byte_ptr" = getelementptr i8, i8* %"stream_pixelsOut_arena_bytes", i32 %"stream_pixelsOut_byte_offset"
  %".20" = bitcast i8* %"stream_pixelsOut_r_byte_ptr" to float*
  %"stream_pixelsOut_val" = load float, float* %".20"
  %".21" = insertvalue %"struct.Pixel" undef, float %"stream_pixelsOut_val", 0
  %"stream_pixelsOut_byte_index.1" = mul i32 %"idx", 4
  %"stream_pixelsOut_byte_offset.1" = add i32 16384, %"stream_pixelsOut_byte_index.1"
  %"stream_pixelsOut_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsOut_g_byte_ptr" = getelementptr i8, i8* %"stream_pixelsOut_arena_bytes.1", i32 %"stream_pixelsOut_byte_offset.1"
  %".22" = bitcast i8* %"stream_pixelsOut_g_byte_ptr" to float*
  %"stream_pixelsOut_val.1" = load float, float* %".22"
  %".23" = insertvalue %"struct.Pixel" %".21", float %"stream_pixelsOut_val.1", 1
  %"stream_pixelsOut_byte_index.2" = mul i32 %"idx", 4
  %"stream_pixelsOut_byte_offset.2" = add i32 20480, %"stream_pixelsOut_byte_index.2"
  %"stream_pixelsOut_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsOut_b_byte_ptr" = getelementptr i8, i8* %"stream_pixelsOut_arena_bytes.2", i32 %"stream_pixelsOut_byte_offset.2"
  %".24" = bitcast i8* %"stream_pixelsOut_b_byte_ptr" to float*
  %"stream_pixelsOut_val.2" = load float, float* %".24"
  %".25" = insertvalue %"struct.Pixel" %".23", float %"stream_pixelsOut_val.2", 2
  store %"struct.Pixel" %".25", %"struct.Pixel"* %"route_pixelsOut_out_slot"
  %"uniform_gain_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"uniform_gain_value_byte_ptr" = getelementptr i8, i8* %"uniform_gain_arena_bytes", i32 24576
  %".27" = bitcast i8* %"uniform_gain_value_byte_ptr" to float*
  %"uniform_gain_val" = load float, float* %".27"
  call void @"shader_Brighten"(%"struct.Pixel" %".15", %"struct.Pixel"* %"route_pixelsOut_out_slot", float %"uniform_gain_val")
  %"route_pixelsOut_out_value" = load %"struct.Pixel", %"struct.Pixel"* %"route_pixelsOut_out_slot"
  %".29" = extractvalue %"struct.Pixel" %"route_pixelsOut_out_value", 0
  %"stream_pixelsOut_byte_index.3" = mul i32 %"idx", 4
  %"stream_pixelsOut_byte_offset.3" = add i32 12288, %"stream_pixelsOut_byte_index.3"
  %"stream_pixelsOut_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsOut_r_byte_ptr.1" = getelementptr i8, i8* %"stream_pixelsOut_arena_bytes.3", i32 %"stream_pixelsOut_byte_offset.3"
  %".30" = bitcast i8* %"stream_pixelsOut_r_byte_ptr.1" to float*
  store float %".29", float* %".30"
  %".32" = extractvalue %"struct.Pixel" %"route_pixelsOut_out_value", 1
  %"stream_pixelsOut_byte_index.4" = mul i32 %"idx", 4
  %"stream_pixelsOut_byte_offset.4" = add i32 16384, %"stream_pixelsOut_byte_index.4"
  %"stream_pixelsOut_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsOut_g_byte_ptr.1" = getelementptr i8, i8* %"stream_pixelsOut_arena_bytes.4", i32 %"stream_pixelsOut_byte_offset.4"
  %".33" = bitcast i8* %"stream_pixelsOut_g_byte_ptr.1" to float*
  store float %".32", float* %".33"
  %".35" = extractvalue %"struct.Pixel" %"route_pixelsOut_out_value", 2
  %"stream_pixelsOut_byte_index.5" = mul i32 %"idx", 4
  %"stream_pixelsOut_byte_offset.5" = add i32 20480, %"stream_pixelsOut_byte_index.5"
  %"stream_pixelsOut_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_pixelsOut_b_byte_ptr.1" = getelementptr i8, i8* %"stream_pixelsOut_arena_bytes.5", i32 %"stream_pixelsOut_byte_offset.5"
  %".36" = bitcast i8* %"stream_pixelsOut_b_byte_ptr.1" to float*
  store float %".35", float* %".36"
  %"idx_next" = add i32 %"idx", 1
  store i32 %"idx_next", i32* %"Brighten_idx"
  br label %"route_Brighten_cond"
route_Brighten_exit:
  ret void
}
