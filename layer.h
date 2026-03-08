#pragma once

#include "node.h"


ExecNode *layer_create_input(Arena *arena, usz *counter, usz num_examples, usz feature_size);
ExecNode *layer_create_target(Arena *arena, usz *counter, usz num_examples, usz label_size);

ExecNode *layer_create_relu(Arena *arena, usz *counter, ExecNode *input);

ExecNode *layer_create_linear(Arena *arena, usz *counter, ExecNode *input, usz in_dim, usz out_dim);

