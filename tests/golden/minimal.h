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

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Pixel {
    float r;
    float g;
    float b;
});

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Arena {
    float stream_pixelsIn_r[1024];
    float stream_pixelsIn_g[1024];
    float stream_pixelsIn_b[1024];
    float stream_pixelsOut_r[1024];
    float stream_pixelsOut_g[1024];
    float stream_pixelsOut_b[1024];
    float uniform_gain_value;
});

#define LOCKSTEP_ARENA_BYTES 24580
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
#define LOCKSTEP_OFFSET_STREAM_PIXELSIN 0
#define LOCKSTEP_OFFSET_STREAM_PIXELSOUT 12288
#define LOCKSTEP_OFFSET_UNIFORM_GAIN 24576
#define LOCKSTEP_OFFSET_STREAM_PIXELSIN_R 0
#define LOCKSTEP_OFFSET_STREAM_PIXELSIN_G 4096
#define LOCKSTEP_OFFSET_STREAM_PIXELSIN_B 8192
#define LOCKSTEP_OFFSET_STREAM_PIXELSOUT_R 12288
#define LOCKSTEP_OFFSET_STREAM_PIXELSOUT_G 16384
#define LOCKSTEP_OFFSET_STREAM_PIXELSOUT_B 20480
#define LOCKSTEP_CAPACITY_STREAM_PIXELSIN 1024
#define LOCKSTEP_CAPACITY_STREAM_PIXELSOUT 1024

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
