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

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Reading {
    int32_t sensorId;
    float value;
    uint8_t valid;
});

LOCKSTEP_PACKED_STRUCT(struct Lockstep_Arena {
    int32_t stream_readingsRaw_sensorId[4096];
    float stream_readingsRaw_value[4096];
    uint8_t stream_readingsRaw_valid[4096];
    int32_t stream_readingsValid_sensorId[4096];
    float stream_readingsValid_value[4096];
    uint8_t stream_readingsValid_valid[4096];
    int32_t stream_readingsAggregated_sensorId[4096];
    float stream_readingsAggregated_value[4096];
    uint8_t stream_readingsAggregated_valid[4096];
    float accum_total_value[4096];
});

#define LOCKSTEP_ARENA_BYTES 126976
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
#define LOCKSTEP_OFFSET_STREAM_READINGSRAW 0
#define LOCKSTEP_OFFSET_STREAM_READINGSVALID 36864
#define LOCKSTEP_OFFSET_STREAM_READINGSAGGREGATED 73728
#define LOCKSTEP_OFFSET_ACCUM_TOTAL 110592
#define LOCKSTEP_OFFSET_STREAM_READINGSRAW_SENSORID 0
#define LOCKSTEP_OFFSET_STREAM_READINGSRAW_VALUE 16384
#define LOCKSTEP_OFFSET_STREAM_READINGSRAW_VALID 32768
#define LOCKSTEP_OFFSET_STREAM_READINGSVALID_SENSORID 36864
#define LOCKSTEP_OFFSET_STREAM_READINGSVALID_VALUE 53248
#define LOCKSTEP_OFFSET_STREAM_READINGSVALID_VALID 69632
#define LOCKSTEP_OFFSET_STREAM_READINGSAGGREGATED_SENSORID 73728
#define LOCKSTEP_OFFSET_STREAM_READINGSAGGREGATED_VALUE 90112
#define LOCKSTEP_OFFSET_STREAM_READINGSAGGREGATED_VALID 106496
#define LOCKSTEP_CAPACITY_STREAM_READINGSRAW 4096
#define LOCKSTEP_CAPACITY_STREAM_READINGSVALID 4096
#define LOCKSTEP_CAPACITY_STREAM_READINGSAGGREGATED 4096

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
