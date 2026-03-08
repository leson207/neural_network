#include "node.h"

#include <math.h>
#include <assert.h>
#include <memory.h>
#include <stdio.h>


static ExecNode * exec_node_create(
    Arena *arena,
    usz id,
    usz rows,
    usz cols,
    u32 flags,
    ExecNode *a,
    ExecNode *b,
    u0 (*fwd)(ExecNode *node),
    u0 (*bwd)(ExecNode *node)
)
{
    if ((a && (a->flags & EXEC_FLAG_REQUIRES_GRAD)) ||
        (b && (b->flags & EXEC_FLAG_REQUIRES_GRAD)))
    {
        flags|=EXEC_FLAG_REQUIRES_GRAD;
    }

    ExecNode *n=PUSH_STRUCT(arena, ExecNode);

    n->index=id;
    n->flags=flags;

    n->val=matrix_create(arena, rows, cols);

    if(flags&EXEC_FLAG_REQUIRES_GRAD) n->grad=matrix_create(arena, rows, cols);
    else n->grad = NULL;

    n->input[0]=a;
    n->input[1]=b;
    n->num_inputs=(a!=NULL)+(b!=NULL);

    n->forward =fwd;
    n->backward=bwd;

    return n;
}

ExecNode *exec_node_create_param(
    Arena *arena,
    usz id,
    usz rows,
    usz cols,
    u32 flags
)
{
    return exec_node_create(
        arena, id,
        rows,
        cols,
        flags,
        NULL, NULL,
        NULL, NULL
    );
}

static u0 exec_op_relu_forward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    matrix_relu(node->val, a->val);
}
static u0 exec_op_relu_backward(ExecNode *node)
{
    Matrix *restrict out=node->input[0]->grad;
    Matrix *restrict in=node->input[0]->val;
    Matrix *restrict grad=node->grad;

    assert(out->rows==in->rows && out->cols==in->cols);
    assert(out->rows==grad->rows && out->cols==grad->cols);

    for(usz i=0; i<out->size; ++i) out->data[i]+=(in->data[i]>0.0f)*grad->data[i];
}
ExecNode *exec_node_create_relu(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
)
{
    assert(a);
    assert(!b);

    return exec_node_create(
        arena, id,
        a->val->rows,
        a->val->cols,
        flags,
        a, b,
        exec_op_relu_forward,
        exec_op_relu_backward
    );
}

static u0 exec_op_softmax_forward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    matrix_softmax(node->val, a->val);
}
static u0 exec_op_softmax_backward(ExecNode *node)
{
    // TODO: can be better
    Matrix *restrict out=node->input[0]->grad;
    Matrix *restrict softmax_out=node->input[0]->val;
    Matrix *restrict grad=node->grad;

    ArenaTemp scratch=arena_get_scratch(NULL, 0);

    usz size=MAX(softmax_out->rows, softmax_out->cols);
    Matrix* jacobian = matrix_create(scratch.arena, size, size);

    for(usz i=0; i<size; ++i)
    {
        for(usz j=0; j<size; ++j)
        {
            jacobian->data[i*size+j]=softmax_out->data[i]*((i==j)-softmax_out->data[j]);
        }
    }
    matrix_mul(out, jacobian, grad, 0, 0, 0);

    arena_release_scratch(scratch);
}
ExecNode *exec_node_create_softmax(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
)
{
    assert(a);
    assert(!b);

    return exec_node_create(
        arena, id,
        a->val->rows,
        a->val->cols,
        flags,
        a, b,
        exec_op_softmax_forward,
        exec_op_softmax_backward
    );
}

static u0 exec_op_add_forward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    matrix_add(node->val, a->val, b->val, 0);
}
static u0 exec_op_add_backward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    if(a->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_add(a->grad, a->grad, node->grad, 0);
    if(b->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_add(b->grad, b->grad, node->grad, 0);
}
ExecNode *exec_node_create_add(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
)
{
    assert(a);
    assert(b);
    assert(a->val->rows==b->val->rows);
    assert(a->val->cols==b->val->cols);

    return exec_node_create(
        arena, id,
        a->val->rows,
        a->val->cols,
        flags,
        a, b,
        exec_op_add_forward,
        exec_op_add_backward
    );
}

static u0 exec_op_add_bias_forward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    matrix_add(node->val, a->val, b->val, 1);
}
static u0 exec_op_add_bias_backward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    if(a->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_add(a->grad, a->grad, node->grad, 0);
    if(b->flags & EXEC_FLAG_REQUIRES_GRAD)
    {
        usz batch_size=node->val->rows;
        usz n_features=node->val->cols;
        for(usz j=0; j<n_features; j++)
        {
            f32 sum=0;
            for(usz i=0; i<batch_size; i++) sum+=node->grad->data[i*n_features+j];
            b->grad->data[j]+=sum;
        }
    }
}
ExecNode *exec_node_create_add_bias(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
)
{
    assert(a);
    assert(b);

    assert(a->val->rows%b->val->rows==0);
    assert(a->val->cols==b->val->cols);

    return exec_node_create(
        arena, id,
        a->val->rows,
        a->val->cols,
        flags,
        a, b,
        exec_op_add_bias_forward,
        exec_op_add_bias_backward
    );
}

static u0 exec_op_sub_forward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    matrix_sub(node->val, a->val, b->val, 1);
}
static u0 exec_op_sub_backward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    if(a->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_sub(a->grad, a->grad, node->grad, 1);
    if(b->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_sub(b->grad, b->grad, node->grad, 1);
}
ExecNode *exec_node_create_sub(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
)
{
    assert(a);
    assert(b);
    assert(a->val->rows==b->val->rows);
    assert(a->val->cols==b->val->cols);

    return exec_node_create(
        arena, id,
        a->val->rows,
        a->val->cols,
        flags,
        a, b,
        exec_op_sub_forward,
        exec_op_sub_backward
    );
}


static u0 exec_op_mul_forward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    matrix_mul(node->val, a->val, b->val, 1, 0, 0);
}
static u0 exec_op_mul_backward(ExecNode *node)
{
    ExecNode *restrict a=node->input[0];
    ExecNode *restrict b=node->input[1];

    if(a->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_mul(a->grad, node->grad, b->val, 0, 0, 1);
    if(b->flags & EXEC_FLAG_REQUIRES_GRAD) matrix_mul(b->grad, a->val, node->grad, 0, 1, 0);
}
ExecNode *exec_node_create_mul(
    Arena *arena,
    usz id,
    ExecNode *a,
    ExecNode *b,
    u32 flags
)
{
    assert(a);
    assert(b);
    assert(a->val->cols==b->val->rows);

    return exec_node_create(
        arena, id,
        a->val->rows,
        b->val->cols,
        flags,
        a, b,
        exec_op_mul_forward,
        exec_op_mul_backward
    );
}

static u0 exec_op_cross_entropy_with_logits_forward(ExecNode *node)
{
    Matrix *restrict logits=node->input[0]->val;
    Matrix *restrict target=node->input[1]->val;

    assert(logits->rows==target->rows);
    assert(logits->cols==target->cols);

    usz batch_size=logits->rows;
    usz label_size=logits->cols;

    f32 total_loss=0.0f;
    for(usz i=0; i<batch_size; ++i)
    {
        f32 *z=logits->data + i*label_size;
        f32 *y=target->data + i*label_size;

        f32 m=z[0];
        for(usz j=1; j<label_size; ++j) m=MAX(m, z[j]);

        f32 sum=0.0f;
        for(usz j=0; j<label_size; ++j) sum+=expf(z[j]-m);

        f32 log_sum_exp=m+logf(sum);
        for(usz j=0; j<label_size; ++j) total_loss+=-y[j]*(z[j]-log_sum_exp);
    }

    node->val->data[0]=total_loss/(f32)batch_size;
}
static u0 exec_op_cross_entropy_with_logits_backward(ExecNode *node)
{
    Matrix *restrict logits=node->input[0]->val;
    Matrix *restrict target=node->input[1]->val;
    Matrix *restrict grad=node->input[0]->grad;

    assert(logits->rows == target->rows);
    assert(logits->cols == target->cols);
    assert(grad->rows == logits->rows && grad->cols == logits->cols);

    usz batch_size = logits->rows;
    usz label_size = logits->cols;

    for(usz i=0; i<batch_size; ++i)
    {
        f32 *z = logits->data + i*label_size;
        f32 *y = target->data + i*label_size;
        f32 *dz = grad->data + i*label_size;

        float m=z[0];
        for(usz j=1; j<label_size; ++j) if(z[j]>m) m=z[j];

        f32 sum=0.0f;
        for(usz j=0; j<label_size; ++j) sum+=expf(z[j]-m);

        for(usz j=0; j<label_size; ++j)
        {
            f32 soft=expf(z[j]-m)/sum;
            dz[j]=(soft-y[j])/(f32)batch_size;
            // dz[j]=node->grad->data[0]*(soft-y[j])/(f32)batch_size;
        }
    }
}
ExecNode *exec_node_create_cross_entropy_with_logits(
    Arena *arena,
    usz id,
    ExecNode *logits,
    ExecNode *target,
    u32 flags
)
{
    assert(logits);
    assert(target);
    assert(logits->val->rows == target->val->rows);
    assert(logits->val->cols == target->val->cols);

    return exec_node_create(
        arena,
        id,
        1, 1,
        flags,
        logits,
        target,
        exec_op_cross_entropy_with_logits_forward,
        exec_op_cross_entropy_with_logits_backward
    );
}
