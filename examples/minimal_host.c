#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lockstep_generated.h"

int main(void) {
    uint8_t arena_bytes[LOCKSTEP_ARENA_BYTES];
    memset(arena_bytes, 0, sizeof(arena_bytes));

    struct Lockstep_Arena* arena = (struct Lockstep_Arena*)arena_bytes;

    arena->stream_particles.pos = 10.0f;
    arena->stream_particles.vel = 2.0f;
    arena->uniform_dt = 0.5f;

    Lockstep_Tick(arena);

    printf("After tick: pos=%.2f vel=%.2f dt=%.2f\n",
           arena->stream_particles.pos,
           arena->stream_particles.vel,
           arena->uniform_dt);
    return 0;
}
