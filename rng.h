#pragma once

#include "base.h"

// PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation

typedef struct Rng Rng;

u32 rng_rand(u0);
u0 rng_seed(u64 init_state, u64 init_seq);
f32 rng_randf(u0);
