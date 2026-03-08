#include "rng.h"


struct Rng
{
    u64 state;
    u64 inc;
};

static Rng s_rng = {
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL,
};

static u32 rng_rand_r(Rng *rng)
{
    u64 old_state=rng->state;
    rng->state=old_state*6364136223846793005ULL+ rng->inc;

    u32 rotation=old_state>>59u;
    u32 xor_shifted=((old_state>>18u)^old_state)>>27u;

    return (xor_shifted>>rotation) | (xor_shifted<<((-rotation)&31));
}

u32 rng_rand(u0)
{
    return rng_rand_r(&s_rng);
}

static u0 rng_seed_r(Rng *rng, u64 init_state, u64 init_seq)
{
    rng->state=0U;
    rng->inc=(init_seq<<1u)|1u;
    rng_rand_r(rng);

    rng->state+=init_state;
    rng_rand_r(rng);

    return;
}

u0 rng_seed(u64 init_state, u64 init_seq)
{
    rng_seed_r(&s_rng, init_state, init_seq);

    return;
}

static f32 rng_randf_r(Rng *rng)
{
    return (f32)rng_rand_r(rng)/(f32)UINT32_MAX;
}

f32 rng_randf(u0)
{
    return rng_randf_r(&s_rng);
}
