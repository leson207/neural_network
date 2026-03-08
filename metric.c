#include "metric.h"

#include <assert.h>


u32 accuracy(Matrix *a, Matrix *b)
{
    assert(a->rows==b->rows && a->cols==b->cols);

    u32 cnt=0;
    for(usz i=0; i<a->rows; ++i)
    {
        usz p_a=0;
        for(usz j=1; j<a->cols; ++j) if(a->data[i*a->cols+j]>a->data[i*a->cols+p_a]) p_a=j;

        usz p_b=0;
        for(usz j=1; j<b->cols; ++j) if(b->data[i*b->cols+j]>b->data[i*b->cols+p_b]) p_b=j;

        if(p_a==p_b) ++cnt;
    }

    return cnt;
}
