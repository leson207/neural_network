#include "matrix.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>


Matrix *matrix_create(Arena *arena, usz rows, usz cols)
{
    Matrix *matrix=PUSH_STRUCT(arena, Matrix);

    matrix->rows=rows;
    matrix->cols=cols;
    matrix->size=rows*cols;
    matrix->data=PUSH_ARRAY(arena, f32, rows*cols);

    return matrix;
}

Matrix *matrix_load(Arena *arena, usz rows, usz cols, const char *filename)
{
    Matrix *matrix=matrix_create(arena, rows, cols);

    FILE *f=fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    u64 size=ftell(f);
    fseek(f, 0, SEEK_SET);

    size=MIN(size, sizeof(f32)*rows*cols);
    fread(matrix->data, 1, size, f);

    fclose(f);

    return matrix;
}

u0 matrix_fill_constant(Matrix *out, f32 value)
{
    for(usz i=0; i<out->size; ++i) out->data[i]=value;
}

u0 matrix_fill_random(Matrix *out, f32 low, f32 high)
{
    for(usz i=0; i<out->size; ++i) out->data[i]=rng_randf()*(high-low)+low;
}

u0 matrix_clear(Matrix *out)
{
    memset(out->data, 0, sizeof(f32)*out->size);
}

f32 matrix_sum(Matrix *matrix)
{
    f32 s=0.0f;
    for(usz i=0; i<matrix->size; ++i) s+=matrix->data[i];

    return s;
}

usz matrix_argmax(Matrix *matrix)
{
    usz p=0;
    for(usz i=1; i<matrix->size; ++i) if(matrix->data[i]>matrix->data[p]) p=i;

    return p;
}

u0 matrix_scale(Matrix *out, f32 scale)
{
    for(usz i=0; i<out->size; ++i) out->data[i]*=scale;
}

u0 matrix_relu(Matrix *out, const Matrix *in)
{
    assert(out->rows==in->rows && out->cols==in->cols);

    for(usz i=0; i<out->size; ++i) out->data[i]=MAX(0, in->data[i]);
}

u0 matrix_softmax(Matrix *out, const Matrix *in)
{
    assert(out->rows==in->rows && out->cols==in->cols);

    f32 sum=0.0f;
    for(usz i=0; i<out->size; ++i)
    {
        out->data[i]=expf(in->data[i]);
        sum+=out->data[i];
    }
    matrix_scale(out, 1.0f/sum);
}

u0 matrix_add(Matrix *out, Matrix *a, Matrix *b, b8 broadcast)
{
    assert(out->rows==a->rows && out->cols==a->cols);

    if(broadcast)
    {
        assert(out->rows%b->rows==0 && out->cols==b->cols);

        for(usz i=0; i<a->rows; i+=b->rows)
        {
            for(usz j=0; j<b->size; ++j) out->data[i*out->cols+j]=a->data[i*a->cols+j]+b->data[j];
        }
    }
    else
    {
        assert(out->rows==b->rows && out->cols==b->cols);
        for(usz i=0; i<out->size; ++i) out->data[i]=a->data[i]+b->data[i];
    }
}

u0 matrix_sub(Matrix *out, Matrix *a, Matrix *b, b8 broadcast)
{
    assert(out->rows==a->rows && out->cols==a->cols);

    if(broadcast)
    {
        assert(out->rows%b->rows==0 && out->cols==b->cols);
        for(usz i=0; i<a->rows; i+=b->rows)
        {
            for(usz j=0; j<b->size; ++j) out->data[i*out->cols+j]=a->data[i*a->cols+j]-b->data[j];
        }
    }
    else
    {
        assert(out->rows==b->rows && out->cols==b->cols);
        for(usz i=0; i<out->size; ++i) out->data[i]=a->data[i]-b->data[i];
    }
}

u0 _matrix_mul_nn(Matrix *restrict out, const Matrix *restrict a, const Matrix *restrict b)
{
    assert(out && a && b);
    assert(a->cols==b->rows);
    assert(out->rows==a->rows);
    assert(out->cols==b->cols);

    usz M=out->rows;
    usz N=a->cols;
    usz P=out->cols;

    for(usz i=0; i<M; ++i)
    {
        for(usz j=0; j<N; ++j)
        {
            f32 x = a->data[i*N + j];

            for(usz k=0; k<P; ++k)
                out->data[i*P + k]+=
                    a->data[i*N + j]*
                    b->data[j*P + k];
        }
    }
}

u0 _matrix_mul_nt(Matrix *restrict out, const Matrix *restrict a, const Matrix *restrict b)
{
    assert(out && a && b);
    assert(a->cols==b->cols);
    assert(out->rows==a->rows);
    assert(out->cols==b->rows);

    usz M=out->rows;
    usz N=a->cols;
    usz P=out->cols;

    for(usz i=0; i<M; ++i)
    {
        // f32 sum = 0; instead of zero out
        for(usz k=0; k<P; ++k)
        {
            for(usz j=0; j<N; ++j)
                out->data[i*P + k]+=
                    a->data[i*N + j]*
                    b->data[k*N + j];
        }
    }
}

u0 _matrix_mul_tn(Matrix *restrict out, const Matrix *restrict a, const Matrix *restrict b)
{
    assert(out && a && b);
    assert(a->rows==b->rows);
    assert(out->rows==a->cols);
    assert(out->cols==b->cols);

    usz M=out->rows;
    usz N=a->rows;
    usz P=out->cols;

    for(usz i=0; i<M; ++i)
    {
        for(usz j=0; j<N; ++j)
        {
            f32 x = a->data[j*M + i];

            for(usz k=0; k<P; ++k)
                out->data[i*P + k]+=
                    a->data[j*M + i]*
                    b->data[j*P + k];
        }
    }
}

u0 _matrix_mul_tt(Matrix *restrict out, const Matrix *restrict a, const Matrix *restrict b)
{
    assert(out && a && b);
    assert(a->rows==b->cols);
    assert(out->rows==a->cols);
    assert(out->cols==b->rows);

    usz M=out->rows;
    usz N=a->rows;
    usz P=out->cols;

    for(usz i=0; i<M; ++i)
    {
        for(usz j=0; j<N; ++j)
        {
            f32 x = a->data[i*N + j];

            for(usz k=0; k<P; ++k)
                out->data[i*P + k]+=
                    a->data[j*M + i]*
                    b->data[k*N + j];
        }
    }
}

u0 matrix_mul(Matrix *out, Matrix *a, Matrix *b, b8 zero_out, b8 transpose_a, b8 transpose_b)
{
    if(zero_out) matrix_clear(out);

    u32 transpose=(transpose_a<<1)|transpose_b;
    switch(transpose)
    {
        case 0b00: _matrix_mul_nn(out, a, b); break;
        case 0b01: _matrix_mul_nt(out, a, b); break;
        case 0b10: _matrix_mul_tn(out, a, b); break;
        case 0b11: _matrix_mul_tt(out, a, b); break;
    }
}

u0 matrix_cross_entropy(Matrix *out, const Matrix *a, const Matrix *b)
{
    assert(out->rows==a->rows && out->cols==a->cols);
    assert(out->rows==b->rows && out->cols==b->cols);

    for(usz i=0; i<out->size; ++i) out->data[i]=a->data[i]==0.0f ? 0.0f : a->data[i]*-logf(b->data[i]);
}
