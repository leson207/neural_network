#include "loss.h"


ExecNode *loss_create_cross_entropy_with_logits(Arena *arena, usz *counter, ExecNode *logits, ExecNode *target)
{
    ExecNode *cost=exec_node_create_cross_entropy_with_logits(
        arena,
        (*counter)++,
        logits,
        target,
        EXEC_FLAG_REQUIRES_GRAD|EXEC_FLAG_COST
    );

    return cost;
}

