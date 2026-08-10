/*
 * Reference implementations of Lockstep's branchless built-in intrinsics.
 *
 * The compiler emits these as external `declare`s in the generated LLVM IR
 * (`pure_step`, `pure_mix`, ...).  Real host applications link against their
 * own optimized versions; the native benchmark harness links this portable
 * C reference so that every workload -- including ones that use straight-line
 * selectors -- produces a runnable executable.
 */

static float lockstep_stepf(float edge, float x) {
    return x < edge ? 0.0f : 1.0f;
}

float pure_step(float edge, float x) {
    return lockstep_stepf(edge, x);
}

float pure_mix(float a, float b, float t) {
    return a + (b - a) * t;
}

float pure_clamp(float x, float min_value, float max_value) {
    const float lo = x < min_value ? min_value : x;
    return lo > max_value ? max_value : lo;
}

float pure_max(float x, float y) {
    return x > y ? x : y;
}

float pure_min(float x, float y) {
    return x < y ? x : y;
}

float pure_abs(float x) {
    return x < 0.0f ? -x : x;
}

float pure_sign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

float pure_smoothstep(float edge0, float edge1, float x) {
    const float span = edge1 - edge0;
    float t = span == 0.0f ? 0.0f : (x - edge0) / span;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * t * (3.0f - 2.0f * t);
}
