; ModuleID = "lockstep"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%"struct.Particle" = type {i32, float, float, float, float, float}
%"struct.Lockstep_Arena" = type {[2048 x i32], [2048 x float], [2048 x float], [2048 x float], [2048 x float], [2048 x float], [2048 x i32], [2048 x float], [2048 x float], [2048 x float], [2048 x float], [2048 x float], [2048 x float]}
declare float @"pure_step"(float %"edge", float %"x")

declare float @"pure_mix"(float %"a", float %"b", float %"t")

declare float @"pure_clamp"(float %"x", float %"min_value", float %"max_value")

declare float @"pure_max"(float %"x", float %"y")

declare float @"pure_min"(float %"x", float %"y")

declare float @"pure_abs"(float %"x")

declare float @"pure_sign"(float %"x")

declare float @"pure_smoothstep"(float %"edge0", float %"edge1", float %"x")

define void @"shader_IntegrateAndAccumulate"(%"struct.Particle" %"src", %"struct.Particle"* %"dst", float* %"kineticEnergy")
{
entry:
  %"src.1" = alloca %"struct.Particle"
  store %"struct.Particle" %"src", %"struct.Particle"* %"src.1"
  %"dst.1" = alloca %"struct.Particle"*
  store %"struct.Particle"* %"dst", %"struct.Particle"** %"dst.1"
  %"kineticEnergy.1" = alloca float*
  store float* %"kineticEnergy", float** %"kineticEnergy.1"
  %"speed2" = alloca float
  store float 0.0, float* %"speed2"
  %"src_val" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vx_field" = extractvalue %"struct.Particle" %"src_val", 3
  %"src_val.1" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vx_field.1" = extractvalue %"struct.Particle" %"src_val.1", 3
  %".9" = fmul float %"vx_field", %"vx_field.1"
  %"src_val.2" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vy_field" = extractvalue %"struct.Particle" %"src_val.2", 4
  %"src_val.3" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vy_field.1" = extractvalue %"struct.Particle" %"src_val.3", 4
  %".10" = fmul float %"vy_field", %"vy_field.1"
  %".11" = fadd float %".9", %".10"
  store float %".11", float* %"speed2"
  %"kineticEnergy_val" = load float*, float** %"kineticEnergy.1"
  %"kineticEnergy_ref" = load float, float* %"kineticEnergy_val"
  %"src_val.4" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"mass_field" = extractvalue %"struct.Particle" %"src_val.4", 5
  %".13" = fmul float 0x3fe0000000000000, %"mass_field"
  %"speed2_val" = load float, float* %"speed2"
  %".14" = fmul float %".13", %"speed2_val"
  %".15" = fadd float %"kineticEnergy_ref", %".14"
  %"kineticEnergy_ptr" = load float*, float** %"kineticEnergy.1"
  store float %".15", float* %"kineticEnergy_ptr"
  %"src_val.5" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"id_field" = extractvalue %"struct.Particle" %"src_val.5", 0
  %"dst_ptr" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  %"dst_ref" = load %"struct.Particle", %"struct.Particle"* %"dst_ptr"
  %"set_id" = insertvalue %"struct.Particle" %"dst_ref", i32 %"id_field", 0
  %"dst_ptr.1" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  store %"struct.Particle" %"set_id", %"struct.Particle"* %"dst_ptr.1"
  %"src_val.6" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"px_field" = extractvalue %"struct.Particle" %"src_val.6", 1
  %"src_val.7" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vx_field.2" = extractvalue %"struct.Particle" %"src_val.7", 3
  %".18" = fadd float %"px_field", %"vx_field.2"
  %"dst_ptr.2" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  %"dst_ref.1" = load %"struct.Particle", %"struct.Particle"* %"dst_ptr.2"
  %"set_px" = insertvalue %"struct.Particle" %"dst_ref.1", float %".18", 1
  %"dst_ptr.3" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  store %"struct.Particle" %"set_px", %"struct.Particle"* %"dst_ptr.3"
  %"src_val.8" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"py_field" = extractvalue %"struct.Particle" %"src_val.8", 2
  %"src_val.9" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vy_field.2" = extractvalue %"struct.Particle" %"src_val.9", 4
  %".20" = fadd float %"py_field", %"vy_field.2"
  %"dst_ptr.4" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  %"dst_ref.2" = load %"struct.Particle", %"struct.Particle"* %"dst_ptr.4"
  %"set_py" = insertvalue %"struct.Particle" %"dst_ref.2", float %".20", 2
  %"dst_ptr.5" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  store %"struct.Particle" %"set_py", %"struct.Particle"* %"dst_ptr.5"
  %"src_val.10" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vx_field.3" = extractvalue %"struct.Particle" %"src_val.10", 3
  %".22" = fmul float %"vx_field.3", 0x3fefd70a40000000
  %"dst_ptr.6" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  %"dst_ref.3" = load %"struct.Particle", %"struct.Particle"* %"dst_ptr.6"
  %"set_vx" = insertvalue %"struct.Particle" %"dst_ref.3", float %".22", 3
  %"dst_ptr.7" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  store %"struct.Particle" %"set_vx", %"struct.Particle"* %"dst_ptr.7"
  %"src_val.11" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"vy_field.3" = extractvalue %"struct.Particle" %"src_val.11", 4
  %".24" = fmul float %"vy_field.3", 0x3fefd70a40000000
  %"dst_ptr.8" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  %"dst_ref.4" = load %"struct.Particle", %"struct.Particle"* %"dst_ptr.8"
  %"set_vy" = insertvalue %"struct.Particle" %"dst_ref.4", float %".24", 4
  %"dst_ptr.9" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  store %"struct.Particle" %"set_vy", %"struct.Particle"* %"dst_ptr.9"
  %"src_val.12" = load %"struct.Particle", %"struct.Particle"* %"src.1"
  %"mass_field.1" = extractvalue %"struct.Particle" %"src_val.12", 5
  %"dst_ptr.10" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  %"dst_ref.5" = load %"struct.Particle", %"struct.Particle"* %"dst_ptr.10"
  %"set_mass" = insertvalue %"struct.Particle" %"dst_ref.5", float %"mass_field.1", 5
  %"dst_ptr.11" = load %"struct.Particle"*, %"struct.Particle"** %"dst.1"
  store %"struct.Particle" %"set_mass", %"struct.Particle"* %"dst_ptr.11"
  ret void
}

define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* noalias nocapture %"arena")
{
entry:
  %"IntegrateAndAccumulate_idx" = alloca i32
  store i32 0, i32* %"IntegrateAndAccumulate_idx"
  br label %"route_IntegrateAndAccumulate_cond"
route_IntegrateAndAccumulate_cond:
  %"idx" = load i32, i32* %"IntegrateAndAccumulate_idx"
  %"route_active" = icmp slt i32 %"idx", 2048
  br i1 %"route_active", label %"route_IntegrateAndAccumulate_body", label %"route_IntegrateAndAccumulate_exit"
route_IntegrateAndAccumulate_body:
  %".6" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx", i32 0
  %"route_i32_splat" = shufflevector <4 x i32> %".6", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".7" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.1" = shufflevector <4 x i32> %".7", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp" = icmp sgt <4 x i32> %"route_i32_splat", %"route_i32_splat.1"
  %"route_vec_max" = select  <4 x i1> %"route_vec_max_cmp", <4 x i32> %"route_i32_splat", <4 x i32> %"route_i32_splat.1"
  %"route_i32_lane0" = extractelement <4 x i32> %"route_vec_max", i32 0
  %".8" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0", i32 0
  %"route_i32_splat.2" = shufflevector <4 x i32> %".8", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".9" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 2047, i32 0
  %"route_i32_splat.3" = shufflevector <4 x i32> %".9", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp" = icmp slt <4 x i32> %"route_i32_splat.2", %"route_i32_splat.3"
  %"route_vec_min" = select  <4 x i1> %"route_vec_min_cmp", <4 x i32> %"route_i32_splat.2", <4 x i32> %"route_i32_splat.3"
  %"route_i32_lane0.1" = extractelement <4 x i32> %"route_vec_min", i32 0
  %"stream_particlesIn_byte_index" = mul i32 %"route_i32_lane0.1", 4
  %"stream_particlesIn_byte_offset" = add i32 0, %"stream_particlesIn_byte_index"
  %"stream_particlesIn_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesIn_id_byte_ptr" = getelementptr i8, i8* %"stream_particlesIn_arena_bytes", i32 %"stream_particlesIn_byte_offset"
  %".10" = bitcast i8* %"stream_particlesIn_id_byte_ptr" to i32*
  %"stream_particlesIn_val" = load i32, i32* %".10"
  %".11" = insertvalue %"struct.Particle" undef, i32 %"stream_particlesIn_val", 0
  %"stream_particlesIn_byte_index.1" = mul i32 %"route_i32_lane0.1", 4
  %"stream_particlesIn_byte_offset.1" = add i32 8192, %"stream_particlesIn_byte_index.1"
  %"stream_particlesIn_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesIn_px_byte_ptr" = getelementptr i8, i8* %"stream_particlesIn_arena_bytes.1", i32 %"stream_particlesIn_byte_offset.1"
  %".12" = bitcast i8* %"stream_particlesIn_px_byte_ptr" to float*
  %"stream_particlesIn_val.1" = load float, float* %".12"
  %".13" = insertvalue %"struct.Particle" %".11", float %"stream_particlesIn_val.1", 1
  %"stream_particlesIn_byte_index.2" = mul i32 %"route_i32_lane0.1", 4
  %"stream_particlesIn_byte_offset.2" = add i32 16384, %"stream_particlesIn_byte_index.2"
  %"stream_particlesIn_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesIn_py_byte_ptr" = getelementptr i8, i8* %"stream_particlesIn_arena_bytes.2", i32 %"stream_particlesIn_byte_offset.2"
  %".14" = bitcast i8* %"stream_particlesIn_py_byte_ptr" to float*
  %"stream_particlesIn_val.2" = load float, float* %".14"
  %".15" = insertvalue %"struct.Particle" %".13", float %"stream_particlesIn_val.2", 2
  %"stream_particlesIn_byte_index.3" = mul i32 %"route_i32_lane0.1", 4
  %"stream_particlesIn_byte_offset.3" = add i32 24576, %"stream_particlesIn_byte_index.3"
  %"stream_particlesIn_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesIn_vx_byte_ptr" = getelementptr i8, i8* %"stream_particlesIn_arena_bytes.3", i32 %"stream_particlesIn_byte_offset.3"
  %".16" = bitcast i8* %"stream_particlesIn_vx_byte_ptr" to float*
  %"stream_particlesIn_val.3" = load float, float* %".16"
  %".17" = insertvalue %"struct.Particle" %".15", float %"stream_particlesIn_val.3", 3
  %"stream_particlesIn_byte_index.4" = mul i32 %"route_i32_lane0.1", 4
  %"stream_particlesIn_byte_offset.4" = add i32 32768, %"stream_particlesIn_byte_index.4"
  %"stream_particlesIn_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesIn_vy_byte_ptr" = getelementptr i8, i8* %"stream_particlesIn_arena_bytes.4", i32 %"stream_particlesIn_byte_offset.4"
  %".18" = bitcast i8* %"stream_particlesIn_vy_byte_ptr" to float*
  %"stream_particlesIn_val.4" = load float, float* %".18"
  %".19" = insertvalue %"struct.Particle" %".17", float %"stream_particlesIn_val.4", 4
  %"stream_particlesIn_byte_index.5" = mul i32 %"route_i32_lane0.1", 4
  %"stream_particlesIn_byte_offset.5" = add i32 40960, %"stream_particlesIn_byte_index.5"
  %"stream_particlesIn_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesIn_mass_byte_ptr" = getelementptr i8, i8* %"stream_particlesIn_arena_bytes.5", i32 %"stream_particlesIn_byte_offset.5"
  %".20" = bitcast i8* %"stream_particlesIn_mass_byte_ptr" to float*
  %"stream_particlesIn_val.5" = load float, float* %".20"
  %".21" = insertvalue %"struct.Particle" %".19", float %"stream_particlesIn_val.5", 5
  %".22" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"idx", i32 0
  %"route_i32_splat.4" = shufflevector <4 x i32> %".22", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".23" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 0, i32 0
  %"route_i32_splat.5" = shufflevector <4 x i32> %".23", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_max_cmp.1" = icmp sgt <4 x i32> %"route_i32_splat.4", %"route_i32_splat.5"
  %"route_vec_max.1" = select  <4 x i1> %"route_vec_max_cmp.1", <4 x i32> %"route_i32_splat.4", <4 x i32> %"route_i32_splat.5"
  %"route_i32_lane0.2" = extractelement <4 x i32> %"route_vec_max.1", i32 0
  %".24" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 %"route_i32_lane0.2", i32 0
  %"route_i32_splat.6" = shufflevector <4 x i32> %".24", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %".25" = insertelement <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, i32 2047, i32 0
  %"route_i32_splat.7" = shufflevector <4 x i32> %".25", <4 x i32> <i32 undef, i32 undef, i32 undef, i32 undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %"route_vec_min_cmp.1" = icmp slt <4 x i32> %"route_i32_splat.6", %"route_i32_splat.7"
  %"route_vec_min.1" = select  <4 x i1> %"route_vec_min_cmp.1", <4 x i32> %"route_i32_splat.6", <4 x i32> %"route_i32_splat.7"
  %"route_i32_lane0.3" = extractelement <4 x i32> %"route_vec_min.1", i32 0
  %"route_particlesOut_out_slot" = alloca %"struct.Particle"
  %"stream_particlesOut_byte_index" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset" = add i32 49152, %"stream_particlesOut_byte_index"
  %"stream_particlesOut_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_id_byte_ptr" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes", i32 %"stream_particlesOut_byte_offset"
  %".26" = bitcast i8* %"stream_particlesOut_id_byte_ptr" to i32*
  %"stream_particlesOut_val" = load i32, i32* %".26"
  %".27" = insertvalue %"struct.Particle" undef, i32 %"stream_particlesOut_val", 0
  %"stream_particlesOut_byte_index.1" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.1" = add i32 57344, %"stream_particlesOut_byte_index.1"
  %"stream_particlesOut_arena_bytes.1" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_px_byte_ptr" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.1", i32 %"stream_particlesOut_byte_offset.1"
  %".28" = bitcast i8* %"stream_particlesOut_px_byte_ptr" to float*
  %"stream_particlesOut_val.1" = load float, float* %".28"
  %".29" = insertvalue %"struct.Particle" %".27", float %"stream_particlesOut_val.1", 1
  %"stream_particlesOut_byte_index.2" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.2" = add i32 65536, %"stream_particlesOut_byte_index.2"
  %"stream_particlesOut_arena_bytes.2" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_py_byte_ptr" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.2", i32 %"stream_particlesOut_byte_offset.2"
  %".30" = bitcast i8* %"stream_particlesOut_py_byte_ptr" to float*
  %"stream_particlesOut_val.2" = load float, float* %".30"
  %".31" = insertvalue %"struct.Particle" %".29", float %"stream_particlesOut_val.2", 2
  %"stream_particlesOut_byte_index.3" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.3" = add i32 73728, %"stream_particlesOut_byte_index.3"
  %"stream_particlesOut_arena_bytes.3" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_vx_byte_ptr" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.3", i32 %"stream_particlesOut_byte_offset.3"
  %".32" = bitcast i8* %"stream_particlesOut_vx_byte_ptr" to float*
  %"stream_particlesOut_val.3" = load float, float* %".32"
  %".33" = insertvalue %"struct.Particle" %".31", float %"stream_particlesOut_val.3", 3
  %"stream_particlesOut_byte_index.4" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.4" = add i32 81920, %"stream_particlesOut_byte_index.4"
  %"stream_particlesOut_arena_bytes.4" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_vy_byte_ptr" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.4", i32 %"stream_particlesOut_byte_offset.4"
  %".34" = bitcast i8* %"stream_particlesOut_vy_byte_ptr" to float*
  %"stream_particlesOut_val.4" = load float, float* %".34"
  %".35" = insertvalue %"struct.Particle" %".33", float %"stream_particlesOut_val.4", 4
  %"stream_particlesOut_byte_index.5" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.5" = add i32 90112, %"stream_particlesOut_byte_index.5"
  %"stream_particlesOut_arena_bytes.5" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_mass_byte_ptr" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.5", i32 %"stream_particlesOut_byte_offset.5"
  %".36" = bitcast i8* %"stream_particlesOut_mass_byte_ptr" to float*
  %"stream_particlesOut_val.5" = load float, float* %".36"
  %".37" = insertvalue %"struct.Particle" %".35", float %"stream_particlesOut_val.5", 5
  store %"struct.Particle" %".37", %"struct.Particle"* %"route_particlesOut_out_slot"
  %"accum_kineticEnergy_byte_index" = mul i32 %"idx", 4
  %"accum_kineticEnergy_byte_offset" = add i32 98304, %"accum_kineticEnergy_byte_index"
  %"accum_kineticEnergy_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"accum_kineticEnergy_value_byte_ptr" = getelementptr i8, i8* %"accum_kineticEnergy_arena_bytes", i32 %"accum_kineticEnergy_byte_offset"
  %".39" = bitcast i8* %"accum_kineticEnergy_value_byte_ptr" to float*
  call void @"shader_IntegrateAndAccumulate"(%"struct.Particle" %".21", %"struct.Particle"* %"route_particlesOut_out_slot", float* %".39")
  %"route_particlesOut_out_value" = load %"struct.Particle", %"struct.Particle"* %"route_particlesOut_out_slot"
  %".41" = extractvalue %"struct.Particle" %"route_particlesOut_out_value", 0
  %"stream_particlesOut_byte_index.6" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.6" = add i32 49152, %"stream_particlesOut_byte_index.6"
  %"stream_particlesOut_arena_bytes.6" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_id_byte_ptr.1" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.6", i32 %"stream_particlesOut_byte_offset.6"
  %".42" = bitcast i8* %"stream_particlesOut_id_byte_ptr.1" to i32*
  store i32 %".41", i32* %".42"
  %".44" = extractvalue %"struct.Particle" %"route_particlesOut_out_value", 1
  %"stream_particlesOut_byte_index.7" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.7" = add i32 57344, %"stream_particlesOut_byte_index.7"
  %"stream_particlesOut_arena_bytes.7" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_px_byte_ptr.1" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.7", i32 %"stream_particlesOut_byte_offset.7"
  %".45" = bitcast i8* %"stream_particlesOut_px_byte_ptr.1" to float*
  store float %".44", float* %".45"
  %".47" = extractvalue %"struct.Particle" %"route_particlesOut_out_value", 2
  %"stream_particlesOut_byte_index.8" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.8" = add i32 65536, %"stream_particlesOut_byte_index.8"
  %"stream_particlesOut_arena_bytes.8" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_py_byte_ptr.1" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.8", i32 %"stream_particlesOut_byte_offset.8"
  %".48" = bitcast i8* %"stream_particlesOut_py_byte_ptr.1" to float*
  store float %".47", float* %".48"
  %".50" = extractvalue %"struct.Particle" %"route_particlesOut_out_value", 3
  %"stream_particlesOut_byte_index.9" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.9" = add i32 73728, %"stream_particlesOut_byte_index.9"
  %"stream_particlesOut_arena_bytes.9" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_vx_byte_ptr.1" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.9", i32 %"stream_particlesOut_byte_offset.9"
  %".51" = bitcast i8* %"stream_particlesOut_vx_byte_ptr.1" to float*
  store float %".50", float* %".51"
  %".53" = extractvalue %"struct.Particle" %"route_particlesOut_out_value", 4
  %"stream_particlesOut_byte_index.10" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.10" = add i32 81920, %"stream_particlesOut_byte_index.10"
  %"stream_particlesOut_arena_bytes.10" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_vy_byte_ptr.1" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.10", i32 %"stream_particlesOut_byte_offset.10"
  %".54" = bitcast i8* %"stream_particlesOut_vy_byte_ptr.1" to float*
  store float %".53", float* %".54"
  %".56" = extractvalue %"struct.Particle" %"route_particlesOut_out_value", 5
  %"stream_particlesOut_byte_index.11" = mul i32 %"idx", 4
  %"stream_particlesOut_byte_offset.11" = add i32 90112, %"stream_particlesOut_byte_index.11"
  %"stream_particlesOut_arena_bytes.11" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*
  %"stream_particlesOut_mass_byte_ptr.1" = getelementptr i8, i8* %"stream_particlesOut_arena_bytes.11", i32 %"stream_particlesOut_byte_offset.11"
  %".57" = bitcast i8* %"stream_particlesOut_mass_byte_ptr.1" to float*
  store float %".56", float* %".57"
  %"idx_next" = add i32 %"idx", 1
  store i32 %"idx_next", i32* %"IntegrateAndAccumulate_idx"
  br label %"route_IntegrateAndAccumulate_cond"
route_IntegrateAndAccumulate_exit:
  ret void
}
