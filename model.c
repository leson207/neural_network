#include "model.h"

#include <memory.h>
#include <stdio.h>


u0 model_compile(Arena *arena, Model *model)
{
    ArenaTemp scratch=arena_get_scratch(&arena, 1);

    b8 *visited=PUSH_ARRAY(scratch.arena, b8, model->num_nodes);
    b8 *saved=PUSH_ARRAY(scratch.arena, b8, model->num_nodes);

    usz stack_size=0;
    ExecNode **stack=PUSH_ARRAY(scratch.arena, ExecNode *, model->num_nodes);

    usz out_size=0;
    ExecNode **out=PUSH_ARRAY(scratch.arena, ExecNode *, model->num_nodes);

    stack[stack_size++]=model->output;

    while(stack_size>0)
    {
        ExecNode *cur=stack[stack_size-1];

        if(visited[cur->index])
        {
            if(!saved[cur->index])
            {
                saved[cur->index]=true;
                out[out_size++]=cur;
            }

            --stack_size;

            continue;
        }

        visited[cur->index]=true;

        for(usz i=0; i<cur->num_inputs; ++i)
        {
            ExecNode *input=cur->input[i];
            if(!visited[input->index]) stack[stack_size++]=input;
        }
    }

    model->topo=PUSH_ARRAY_NZ(arena, ExecNode *, out_size);

    model->num_nodes=out_size;
    memcpy(model->topo, out, sizeof(ExecNode *) * out_size);

    for(usz i=0; i<out_size; ++i) printf("%lu ", model->topo[i]->index);
    printf("\n\n");

    arena_release_scratch(scratch);
}

u0 model_forward(Model *model)
{
    usz p=0;
    // skip parameter node
    while(model->topo[p]->index!=0) ++p;
    // skip input node
    ++p;

    for(usz i=p; i<model->num_nodes; ++i)
    {
        ExecNode *cur=model->topo[i];
        cur->forward(cur);
    }
}

u0 model_zero_grad(Model *model)
{
    for(usz i=0; i<model->num_nodes; ++i)
    {
        ExecNode *node=model->topo[i];
        if(node->flags&EXEC_FLAG_REQUIRES_GRAD) matrix_clear(node->grad);
    }
}

u0 model_backward(Model *model)
{
    for(i32 i=model->num_nodes-1; i>=0; --i)
    {
        ExecNode *node=model->topo[i];

        if(node->index==0) break;
        if(!(node->flags&EXEC_FLAG_REQUIRES_GRAD)) continue;

        node->backward(node);
    }
}

u0 model_update(Model *model, f32 learning_rate, usz batch_size)
{
    for(usz i=0; i<model->num_nodes; ++i)
    {
        ExecNode *node=model->topo[i];

        if(node->index==0) break;
        if(!(node->flags&EXEC_FLAG_REQUIRES_GRAD)) continue;

        matrix_scale(node->grad, learning_rate/batch_size);
        matrix_sub(node->val, node->val, node->grad, 0);
    }
}
