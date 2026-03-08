#include "layer.h"
#include "init.h"

#include <stddef.h>


ExecNode *layer_create_input(Arena *arena, usz *counter, usz num_examples, usz feature_size)
{
    ExecNode *node=exec_node_create_param(
        arena,
        (*counter)++,
        num_examples,
        feature_size,
        EXEC_FLAG_INPUT
    );

    return node;
}

ExecNode *layer_create_target(Arena *arena, usz *counter, usz num_examples, usz label_size)
{
    ExecNode *node=exec_node_create_param(
        arena,
        (*counter)++,
        num_examples,
        label_size,
        EXEC_FLAG_TARGET
    );

    return node;
}

ExecNode *layer_create_relu(Arena *arena, usz *counter, ExecNode *input)
{
    ExecNode *a=exec_node_create_relu(
        arena, (*counter)++,
        input, NULL, 0
    );

    return a;
}

ExecNode *layer_create_add(Arena *arena, usz *counter, ExecNode *input, ExecNode *output)
{
    ExecNode *o=exec_node_create_add(
        arena,
        (*counter)++,
        input,
        output,
        EXEC_FLAG_REQUIRES_GRAD
    );

    return o;
}

ExecNode *layer_create_linear(Arena *arena, usz *counter, ExecNode *input, usz in_dim, usz out_dim)
{
    ExecNode *W=exec_node_create_param(
        arena,
        (*counter)++,
        in_dim,
        out_dim,
        EXEC_FLAG_REQUIRES_GRAD | EXEC_FLAG_PARAMETER
    );

    ExecNode *b=exec_node_create_param(
        arena,
        (*counter)++,
        1,
        out_dim,
        EXEC_FLAG_REQUIRES_GRAD | EXEC_FLAG_PARAMETER
    );

    init_random(W->val);

    ExecNode *z=exec_node_create_mul(
        arena, (*counter)++,
        input, W, 0
    );

    ExecNode *a=exec_node_create_add_bias(
        arena, (*counter)++,
        z, b, 0
    );

    return a;
}
