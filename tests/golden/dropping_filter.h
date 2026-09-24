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

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Sample {
    int32_t id;
    float value;
    uint8_t flagged;
});

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Arena {
    int32_t stream_samplesRaw_id[20];
    float stream_samplesRaw_value[20];
    uint8_t stream_samplesRaw_flagged[20];
    int32_t stream_samplesKept_id[20];
    float stream_samplesKept_value[20];
    uint8_t stream_samplesKept_flagged[20];
    int32_t stream_samplesScaled_id[20];
    float stream_samplesScaled_value[20];
    uint8_t stream_samplesScaled_flagged[20];
    float accum_total_value[20];
    float uniform_keptTotal_value;
    uint32_t count_samplesKept_value;
    uint32_t count_samplesScaled_value;
});

#define LOCKSTEP_ARENA_BYTES 632
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
#define LOCKSTEP_OFFSET_STREAM_SAMPLESRAW 0
#define LOCKSTEP_OFFSET_STREAM_SAMPLESKEPT 180
#define LOCKSTEP_OFFSET_STREAM_SAMPLESSCALED 360
#define LOCKSTEP_OFFSET_ACCUM_TOTAL 540
#define LOCKSTEP_OFFSET_UNIFORM_KEPTTOTAL 620
#define LOCKSTEP_OFFSET_COUNT_SAMPLESKEPT 624
#define LOCKSTEP_OFFSET_COUNT_SAMPLESSCALED 628
#define LOCKSTEP_OFFSET_STREAM_SAMPLESRAW_ID 0
#define LOCKSTEP_OFFSET_STREAM_SAMPLESRAW_VALUE 80
#define LOCKSTEP_OFFSET_STREAM_SAMPLESRAW_FLAGGED 160
#define LOCKSTEP_OFFSET_STREAM_SAMPLESKEPT_ID 180
#define LOCKSTEP_OFFSET_STREAM_SAMPLESKEPT_VALUE 260
#define LOCKSTEP_OFFSET_STREAM_SAMPLESKEPT_FLAGGED 340
#define LOCKSTEP_OFFSET_STREAM_SAMPLESSCALED_ID 360
#define LOCKSTEP_OFFSET_STREAM_SAMPLESSCALED_VALUE 440
#define LOCKSTEP_OFFSET_STREAM_SAMPLESSCALED_FLAGGED 520
#define LOCKSTEP_CAPACITY_STREAM_SAMPLESRAW 20
#define LOCKSTEP_CAPACITY_STREAM_SAMPLESKEPT 20
#define LOCKSTEP_CAPACITY_STREAM_SAMPLESSCALED 20

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
