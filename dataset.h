#pragma once

#include "base.h"
#include "arena.h"
#include "matrix.h"


typedef struct Dataset Dataset;

struct Dataset
{
    usz num_samples;
    usz feature_size;
    usz label_size;
    usz *order;

    struct Matrix *feature;
    struct Matrix *label;
};

u0 inspect(f32 *feature, f32 *label);

struct Dataset *dataset_load(
    struct Arena *arena,
    usz num_samples,
    usz feature_size,
    usz label_size,
    const char *feature_file,
    const char *label_file
);

u0 dataset_shuffle(Dataset *ds);
u0 dataset_get(struct Matrix *feature, struct Matrix *label, Dataset *ds, usz idx, usz n);
