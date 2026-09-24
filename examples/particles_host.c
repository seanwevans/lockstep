/*
 * Host for examples/particles.lock.
 *
 * Shows the whole host contract:
 *   - allocate the arena and prime the input stream (SoA columns);
 *   - write every uniform the program reads (initializers in the .lock file
 *     document intent; the host owns the arena, so it sets the values);
 *   - call Lockstep_Tick once per frame;
 *   - read back folded uniforms, and a filtered stream's live row count before
 *     reading its rows.
 *
 * Output is deterministic, so tests/test_examples.py compares it with the
 * simulator.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lockstep_generated.h"

#define N LOCKSTEP_CAPACITY_STREAM_PARTICLES
#define FRAMES 3

int main(void) {
    struct Lockstep_Arena* arena = calloc(1, sizeof(struct Lockstep_Arena));
    if (arena == NULL) {
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        arena->stream_particles_x[i] = (float)(i % 125) * 0.9f;   /* 0 .. 111.6 */
        arena->stream_particles_y[i] = (float)(i % 17) * 0.5f;
        arena->stream_particles_vx[i] = (float)((i % 11) - 5);    /* -5 .. 5 */
        arena->stream_particles_vy[i] = 0.0f;
        arena->stream_particles_mass[i] = 1.0f + (float)(i % 3);
    }
    arena->uniform_dt_value = 0.02f;
    arena->uniform_width_value = 100.0f;

    for (int frame = 1; frame <= FRAMES; ++frame) {
        Lockstep_Tick(arena);

        uint32_t inside = arena->count_tallied_value;
        float mean_x = 0.0f;
        for (uint32_t i = 0; i < inside; ++i) {
            mean_x += arena->stream_tallied_x[i];
        }
        mean_x = inside ? mean_x / (float)inside : 0.0f;

        printf("frame %d: kinetic energy %.3f, inside %u/%d (kept %d), mean x %.3f\n",
               frame,
               (double)arena->uniform_kineticEnergy_value,
               inside,
               N,
               arena->uniform_keptCount_value,
               (double)mean_x);
    }

    free(arena);
    return 0;
}
