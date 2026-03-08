#pragma once

#include "model.h"
#include "dataset.h"


typedef struct Trainer Trainer;

struct Trainer
{
    usz epochs;
    usz batch_size;
    f32 learning_rate;

    usz feature_size;
    usz label_size;

    struct Dataset *train;
    struct Dataset *val;
    struct Dataset *test;

    struct Model *model;
};

Trainer *trainer_create(Arena *arena, usz feature_size, usz label_size);
u0 trainer_create_model(Arena *arena, struct Trainer *trainer);

// u0 _trainer_train(Trainer *trainer);
// u0 _trainer_eval(Trainer *trainer);
// u0 _trainer_test(Trainer *trainer);
u0 trainer_train(Trainer *trainer);
