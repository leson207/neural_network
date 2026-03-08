#pragma once

#include "base.h"
#include "node.h"


typedef struct Model Model;

struct Model
{
    usz num_nodes;
    ExecNode **topo;
    // ExecNode *layers;

    ExecNode *input;
    ExecNode *output;
    ExecNode *target;
    ExecNode *cost;
};

u0 model_compile(Arena *arena, Model *model);
u0 model_forward(Model *model);
u0 model_zero_grad(Model *model);
u0 model_backward(Model *model);
u0 model_update(Model *model, f32 learning_rate, usz batch_size);
