#pragma once

#include "base.h"
#include "matrix.h"


typedef enum {
    EXEC_FLAG_NONE = 0,

    EXEC_FLAG_REQUIRES_GRAD  = (1 << 0),
    EXEC_FLAG_PARAMETER      = (1 << 1),
    EXEC_FLAG_INPUT          = (1 << 2),
    EXEC_FLAG_OUTPUT         = (1 << 3),
    EXEC_FLAG_TARGET         = (1 << 4),
    EXEC_FLAG_COST           = (1 << 5),
} exec_flag;

typedef struct ExecNode ExecNode;

struct ExecNode
{
    usz index;
    u32   flags;

    Matrix *val;
    Matrix *grad;

    usz num_inputs;
    ExecNode *input[2];

    u0 (*forward)(ExecNode *);
    u0 (*backward)(ExecNode *);
};

ExecNode *exec_node_create_param(
    Arena *arena,
    usz id,
    usz rows,
    usz cols,
    u32 flags
);

ExecNode *exec_node_create_relu(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
);

ExecNode *exec_node_create_softmax(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
);

ExecNode *exec_node_create_add(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
);

ExecNode *exec_node_create_add_bias(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
);

ExecNode *exec_node_create_sub(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
);

ExecNode *exec_node_create_mul(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
);

ExecNode *exec_node_create_cross_entropy_with_logits(
    Arena *arena,
    usz id,
    ExecNode *logits,
    ExecNode *target,
    u32 flags
);
