#ifndef LOCKSTEP_GENERATED_H
#define LOCKSTEP_GENERATED_H

#include <stdint.h>
#include <stddef.h>
#ifdef LOCKSTEP_DEBUG_SATURATED_WRITES
#include <stdio.h>
#endif

#if defined(_MSC_VER)
#define LOCKSTEP_PACKED_STRUCT(definition) __pragma(pack(push, 1)) definition __pragma(pack(pop))
#else
#define LOCKSTEP_PACKED_STRUCT(definition) definition __attribute__((packed))
#endif

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Particle {
    int32_t id;
    float px;
    float py;
    float vx;
    float vy;
    float mass;
});

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Arena {
    int32_t stream_particlesIn_id[2048];
    float stream_particlesIn_px[2048];
    float stream_particlesIn_py[2048];
    float stream_particlesIn_vx[2048];
    float stream_particlesIn_vy[2048];
    float stream_particlesIn_mass[2048];
    int32_t stream_particlesOut_id[2048];
    float stream_particlesOut_px[2048];
    float stream_particlesOut_py[2048];
    float stream_particlesOut_vx[2048];
    float stream_particlesOut_vy[2048];
    float stream_particlesOut_mass[2048];
    float accum_kineticEnergy_value[2048];
});

#define LOCKSTEP_ARENA_BYTES 106496
#define LOCKSTEP_SIMD_WIDTH 8
#if defined(__cplusplus) && (__cplusplus >= 201103L)
static_assert(LOCKSTEP_ARENA_BYTES <= SIZE_MAX, "LOCKSTEP_ARENA_BYTES must fit in size_t on the target architecture");
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(LOCKSTEP_ARENA_BYTES <= SIZE_MAX, "LOCKSTEP_ARENA_BYTES must fit in size_t on the target architecture");
#endif
#if defined(__cplusplus) && (__cplusplus >= 201103L)
static_assert(sizeof(struct Lockstep_Arena) == LOCKSTEP_ARENA_BYTES, "Lockstep_Arena size must match LOCKSTEP_ARENA_BYTES");
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(struct Lockstep_Arena) == LOCKSTEP_ARENA_BYTES, "Lockstep_Arena size must match LOCKSTEP_ARENA_BYTES");
#endif
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN 0
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT 49152
#define LOCKSTEP_OFFSET_ACCUM_KINETICENERGY 98304
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN_ID 0
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN_PX 8192
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN_PY 16384
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN_VX 24576
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN_VY 32768
#define LOCKSTEP_OFFSET_STREAM_PARTICLESIN_MASS 40960
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_ID 49152
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_PX 57344
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_PY 65536
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_VX 73728
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_VY 81920
#define LOCKSTEP_OFFSET_STREAM_PARTICLESOUT_MASS 90112
#define LOCKSTEP_CAPACITY_STREAM_PARTICLESIN 2048
#define LOCKSTEP_CAPACITY_STREAM_PARTICLESOUT 2048

#ifndef LOCKSTEP_SATURATED_WRITE_LOG
#define LOCKSTEP_SATURATED_WRITE_LOG(stream_name, index, capacity, saturated_index) \
    fprintf(stderr, "[lockstep] saturated write stream=%s index=%zu capacity=%zu -> %zu\n", \
            (stream_name), (size_t)(index), (size_t)(capacity), (size_t)(saturated_index))
#endif

static inline size_t Lockstep_SaturatedWriteIndex(size_t index, size_t capacity, const char* stream_name) {
    if (capacity == 0) {
        return 0;
    }
    if (index < capacity) {
        return index;
    }
    const size_t saturated_index = capacity - 1;
#ifdef LOCKSTEP_DEBUG_SATURATED_WRITES
    LOCKSTEP_SATURATED_WRITE_LOG(stream_name != NULL ? stream_name : "<unnamed>", index, capacity, saturated_index);
#endif
    return saturated_index;
}

#ifdef __cplusplus
extern "C" {
#endif

void Lockstep_Tick(struct Lockstep_Arena* arena);

#ifdef __cplusplus
}
#endif

#endif
